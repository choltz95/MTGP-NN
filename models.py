import torch
import gpytorch
import torch.nn as nn
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import pandas as pd 
import numpy as np

class GPCNNLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden=64):
        super(GPCNNLSTM, self).__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.predictor_model = CRNN(7, lstm_hidden, n_class = 2, leakyRelu=True).cuda()
        self.interpolation_model = None
        self.params = [
                        {'params': self.predictor_model.parameters()}
                      ]
        #self.optimizer = optim.AdamW(self.predictor_model.parameters(), lr = 0.0001)
        self.optimizer = optim.AdamW(self.params, lr = 0.0001)
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.35,0.65]).cuda(),reduction='mean')
        self.output_transform = nn.Softmax(dim=1)

    def init_interpolation_model(self,input_data):
        x, ind, y = input_data
        self.interpolation_model = MTGPInterpolationModel((x, ind), y, self.likelihood).cuda()
        self.params.append({'params': self.interpolation_model.covar_module.parameters(), 'lr':0.001})
        self.params.append({'params': self.interpolation_model.mean_module.parameters(), 'lr':0.001})
        self.params.append({'params': self.interpolation_model.likelihood.parameters()})


    def update(self, input_data, labels):
        batch_size = len(input_data)
        losses = []
        loss = 0.0
        for j in range(batch_size):
            x, ind, y = input_data[j]

            self.predictor_model.train()
            self.optimizer.zero_grad()

            with gpytorch.settings.detach_test_caches(False), \
                 gpytorch.settings.use_toeplitz(True), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), \
                 gpytorch.settings.max_root_decomposition_size(12):
                self.interpolation_model.set_train_data((x,ind),y,strict=False)
                self.interpolation_model.train()
                #self.interpolation_model.likelihood.train()
                self.interpolation_model.eval()

                sample = []
                x_eval = torch.linspace(0, labels[j].shape[1]-1, labels[j].shape[1]).type(torch.FloatTensor).cuda()
                for ii in range(7):
                    task_idx = torch.full_like(x_eval, dtype=torch.long, fill_value=ii).cuda()
                    gp_output = self.interpolation_model(x_eval, task_idx)
                    f_samples = gp_output.rsample(torch.Size([10]))
                    #f_samples = f_samples.transpose(-2, -1)
                    sample_mean = f_samples.mean(0).squeeze(-1)  # Average over GP sample dimension
                    sample.append(sample_mean)

                    del task_idx

            #print(torch.stack(sample).shape) # d x t
            vital_features = torch.stack(sample).unsqueeze(0).cuda()
            #vital_features = vital_features.permute(0, 2, 1) # b d t

            output = self.predictor_model(vital_features).squeeze(0)
            output = self.output_transform(output)
            loss += self.loss(output, labels[j].squeeze())

            del vital_features
            del x_eval

        loss = loss / batch_size
        losses.append(loss.cpu().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.25)

        """
        task_losses = torch.stack([sepsis_head, duration_head])
        weighted_losses = self.weights * task_losses
        total_weighted_loss = weighted_losses.sum()
        # compute and retain gradients
        total_weighted_loss.backward(retain_graph=True)
        # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
        self.weights.grad = 0.0 * self.weights.grad

        W = list(self.model.mlp_list[-1].parameters())
        norms = []

        for w_i, L_i in zip(self.weights, task_losses):
                gLgW = torch.autograd.grad(L_i, W, retain_graph = True)
                norms.append(torch.norm(w_i * gLgW[0]))

        norms = torch.stack(norms)

        if t ==0:
                self.initial_losses = task_losses.detach()

        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
                # loss ratios \curl{L}(t)
                loss_ratios = task_losses / self.initial_losses
                # inverse training rate r(t)
                inverse_train_rates = loss_ratios / loss_ratios.mean()
                constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        grad_norm_loss = (norms - constant_term).abs().sum()
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        """

        self.optimizer.step()
        del output
        torch.cuda.empty_cache()
        del loss

        return np.mean(losses)

    def predict(self, input_data, seq_length):
        self.interpolation_model.eval()
        self.predictor_model.eval()

        vital_features, lab_features, baseline_features = input_data
        with gpytorch.settings.detach_test_caches(state=True), \
             gpytorch.settings.use_toeplitz(False), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), \
             gpytorch.settings.max_root_decomposition_size(10):
            sample = []
            x_eval = torch.linspace(0, seq_length, seq_length).type(torch.FloatTensor).cuda()
            for ii in range(7):
                task_idx = torch.full_like(x_eval, dtype=torch.long, fill_value=ii).cuda()
                with torch.no_grad():
                    gp_output = self.interpolation_model(x_eval, task_idx)
                    f_samples = gp_output.rsample(torch.Size([10]))
                sample_mean = f_samples.mean(0).squeeze(-1)  # Average over GP sample dimension
                sample.append(sample_mean) 

                del task_idx

        vital_features = torch.stack(sample).unsqueeze(0).cuda()
        output = self.predictor_model(vital_features).squeeze(0)
        output = self.output_transform(output)
        del x_eval
        del vital_features
        #return self.output_transform(self.predictor_model(vital_features.permute(0, 2, 1)).squeeze(0))
        return output

class MTGPInterpolationModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MTGPInterpolationModel, self).__init__(train_x, train_y, likelihood)
        # SKI requires a grid size hyperparameter. This util can help with that

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=7, rank=1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(LSTM, self).__init__()
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        #T, b, h = recurrent.size()
        #t_rec = recurrent.view(T * b, h)
        #output = self.embedding(t_rec)  # [T * b, nOut]
        #output = output.view(T, b, -1)
        output = recurrent

        return output


class CRNN(nn.Module):
    def __init__(self, input_dim, lstm_hidden, n_class, leakyRelu=True):
        super(CRNN, self).__init__()

        self.fc_att = nn.Linear(lstm_hidden, 1)
        self.att_softmax = nn.Softmax(dim=1)
        self.embedding = nn.Linear(lstm_hidden, n_class)

        kernel_size = [3, 3, 3, 3]
        pad_size = [1, 1, 1, 1]
        shift_size = [1, 1, 1, 1]
        in_size = [16, 32, 32, 64]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False, dropout=False):
            nIn = input_dim if i == 0 else in_size[i - 1]
            nOut = in_size[i]
            cnn.add_module('conv{0}'.format(i),
                       nn.Conv1d(nIn, nOut, kernel_size[i], shift_size[i], pad_size[i]))
            if dropout:
                cnn.add_module('dropout{0}'.format(i), nn.Dropout(0.1))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm1d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        convRelu(2)
        convRelu(3, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            LSTM(7, lstm_hidden, lstm_hidden),LSTM(lstm_hidden, lstm_hidden, lstm_hidden))

        #self.register_backward_hook(self.backward_hook)

    def forward(self, input):
        # conv features
        #conv = self.cnn(input) 
        #b, d, t = conv.size()
        #assert t == input.shape[-1], "t must be the same as input"
        #conv = conv.permute(0, 2, 1)  # [t, b, c]

        #print(input.shape)
        #input = 0
        conv = input.permute(2,0,1)
        # rnn features
        output = self.rnn(conv)
        #print('rnn',output.shape)
        #att = self.fc_att(output).squeeze(1)
        #print('att',output.shape)
        #att = self.att_softmax(att)
        #r_att = torch.sum(att.unsqueeze(1) * output, dim=1)
        #output =self.embedding(r_att)
        output =self.embedding(output).squeeze(1)
        #print('attn',output.shape)

        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero