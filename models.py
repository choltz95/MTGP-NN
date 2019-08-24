import torch
import gpytorch
import torch.nn as nn
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import pandas as pd 
import numpy as np

class GPCNNLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden=32):
        super(GPCNNLSTM, self).__init__()
        #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=input_dim)
        #self.interpolation_model = MTGPInterpolationModel(input_dim, train_x, train_y, likelihood)
        self.predictor_model = CRNN(input_dim[1], lstm_hidden, n_class = 2, leakyRelu=True).cuda()
        self.optimizer = optim.AdamW(self.predictor_model.parameters(), lr = 0.0001)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.2,0.8]).cuda(),reduction='mean')
        #self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.output_transform = nn.Softmax(dim=1)
        #self.loss = nn.CrossEntropyLoss(reduction='mean')

    def update(self, input_data, labels):
        #n_batches = int(labels.shape[0]/self.batch_size) 
        batch_size = len(input_data)
        losses = []
        loss = 0.0
        for j in range(batch_size):
            vital_features, lab_features, baseline_features = input_data[j]
            #with gpytorch.settings.detach_test_caches(False):
            #    self.interpolation_model.train()
            #    self.interpolation_model.eval()
            #    gp_output = self.interpolation_model(vital_features)

            self.predictor_model.train()
            self.optimizer.zero_grad()

            #vital_features = gp_output.rsample(torch.Size([10]))
            vital_features = vital_features.permute(0, 2, 1) # b d t

            output = self.predictor_model(vital_features).squeeze(0)
            output = self.output_transform(output)
            #print(output)

            #print('label',labels.shape)
            #print('output',output.shape)
            loss += self.loss(output, labels[j].squeeze())
            #output = self.predictor_model(self.interpolation_model(vital_features))
            #loss = self.loss(output[j*self.batch_size:min((j+1)*self.batch_size, labels.shape[0]-1)], labels[j*self.batch_size:min((j+1)*self.batch_size, labels.shape[0]-1)])

        loss = loss / batch_size
        losses.append(loss.cpu().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.25)

        self.optimizer.step()
        del output
        torch.cuda.empty_cache()
        del loss

        return np.mean(losses)

    def predict(self, input_data):
        self.predictor_model.eval()
        vital_features, lab_features, baseline_features = input_data
        #gp_output = self.interpolation_model(vital_features)
        #vital_features = gp_output.rsample(torch.Size([10]))
        #vital_features = vital_features.transpose(-2,-1)
        return self.output_transform(self.predictor_model(vital_features.permute(0, 2, 1)).squeeze(0))

class MTGPInterpolationModel(gpytorch.models.ExactGP):
    def __init__(self, input_dim, train_x, train_y, likelihood):
        super(MTGPInterpolationModel, self).__init__(train_x, train_y, likelihood)

        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        self.mean_module = gpytorch.means.MultitaskMean()
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)),
            num_dims=input_dim, grid_size=grid_size
        ), num_tasks=input_dim, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 


class LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(LSTM, self).__init__()
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, input_dim, lstm_hidden, n_class, leakyRelu=True):
        super(CRNN, self).__init__()

        kernel_size = [3, 3, 3, 3, 3, 3, 3]
        pad_size = [1, 1, 1, 1, 1, 1, 0]
        shift_size = [1, 1, 1, 1, 1, 1, 1]
        in_size = [16, 32, 32, 64, 512, 512, 512]

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
            LSTM(64, lstm_hidden, n_class))

        #self.register_backward_hook(self.backward_hook)

    def forward(self, input):
        # conv features
        conv = self.cnn(input) 
        b, d, t = conv.size()
        assert t == input.shape[-1], "t must be the same as input"
        conv = conv.permute(0, 2, 1)  # [t, b, c]

        # rnn features
        output = self.rnn(conv)
        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero