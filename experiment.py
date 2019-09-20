import pickle
import sys
import time
from pathlib import Path
from typing import Union

from dataset import Dataset
from models import GPCNNLSTM
from evaluate_sepsis_score import compute_prediction_utility

import fire
from tqdm import tqdm
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

    
def move_to_cuda(input_data): 
    input_data_ = []
    for d in input_data:
        input_data_.append(d.cuda())
    return input_data_

class Experiment:  
    def __init__(self, processing_params=None, model_params=None, training_params=None):
        self.processing_params = processing_params
        self.model_params = model_params
        self.training_params = training_params
        self.batch_size = 32

        self.global_nfp = None
        self.estimator = None
        self.model = None

        self.tr_embeddings_list = []
        self.tr_output_list = []

        self.dataset = None

    @classmethod
    def train(self, training_dict, validation_dict, processing_params=None, model_params=None, training_params=None):

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        model = self(processing_params, model_params, training_params)
        model.model = model
        # Preprocess the data
        print('preprocessing')
        for i, _ in enumerate(tqdm(training_dict['pid_list'], desc='pruning')):
            if training_dict['vital_features_list'][i].shape[0] > 200:
                for key in training_dict:
                    if key == 'pid_list':
                        continue
                    training_dict[key][i] = training_dict[key][i].iloc[training_dict['vital_features_list'][i].shape[0] - 150:]

        training_dict['input_list'] = []

        full_train_not_nan_indexes = []
        full_train_task_indexes = []
        full_train_anchor_values = []

        for i, _ in enumerate(tqdm(training_dict['pid_list'], desc='fillna & normalize')):
            not_nan_indexes = []
            task_indexes = []
            anchor_values = []

            for ii,col in enumerate(training_dict['vital_features_list'][0].columns.tolist()):
                if col == "index":
                    continue
                iii = ii - 1
                training_dict['vital_features_list'][i][col] = np.log(training_dict['vital_features_list'][i][col])
                if training_dict['vital_features_list'][i][col].count() == 0:
                    training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].fillna(0)
                training_dict['vital_features_list'][i][col] = (training_dict['vital_features_list'][i][col] - training_dict['vital_features_list'][i][col].mean())/training_dict['vital_features_list'][i][col].std(ddof=0)
                training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].fillna(0)

                #elif training_dict['vital_features_list'][i][col].count() == 1:
                #    training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].fillna(training_dict['vital_features_list'][i][col].mean())
                #elif training_dict['vital_features_list'][i][col].count() <= 3 :
                #    training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].interpolate(method='linear').interpolate(method='nearest')

                #if training_dict['vital_features_list'][i][col].count() > 3:
                #    training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].interpolate(method='pchip', order=3, axis=0).interpolate(method='nearest')
                #training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].fillna(method='ffill').fillna(method='bfill')
                #training_dict['vital_features_list'][i][col] = training_dict['vital_features_list'][i][col].fillna(0.0)

                not_nan = training_dict['vital_features_list'][i][col].notnull()
                nn_ind = torch.FloatTensor(not_nan[not_nan == True].index.values)
                nn_vals = torch.FloatTensor(training_dict['vital_features_list'][i][col][not_nan == True].values)
                task_ind = torch.full_like(nn_vals, dtype=torch.long, fill_value=iii)

                not_nan_indexes.append(nn_ind)
                task_indexes.append(task_ind)
                anchor_values.append(nn_vals)

                full_train_not_nan_indexes.append(nn_ind)
                full_train_task_indexes.append(task_ind)
                full_train_anchor_values.append(nn_vals)

            full_train_x = torch.cat(not_nan_indexes)
            full_train_i = torch.cat(task_indexes)
            full_train_y = torch.cat(anchor_values)

            training_dict['input_list'].append([full_train_x, full_train_i, full_train_y])

        input_dim = training_dict['vital_features_list'][0].shape
        model.estimator = GPCNNLSTM(input_dim)
        #print('instantiating interpolation model with', len(full_train_not_nan_indexes),'data points')
        init_data = move_to_cuda([full_train_not_nan_indexes[0],full_train_task_indexes[0], full_train_anchor_values[0]])
        model.estimator.init_interpolation_model(init_data)
        del init_data

        loss_iterations = []
        loss_history = []
        tr_util = []
        val_util = []
        # epochs
        for k in tqdm(range(201), desc='epoch iterations'):
            loss_iteration_k = []

            # train on short sequences first
            if k == 0:
                len_sorted_idx = [b[0] for b in sorted(enumerate(training_dict['vital_features_list']), key=lambda i:i[1].shape[0])]
                for key in training_dict:
                    training_dict[key] = [training_dict[key][i] for i in len_sorted_idx]
            else:
                shuff_idx = list(range(len(training_dict['pid_list'])))
                shuffle(shuff_idx)
                for key in training_dict:
                    training_dict[key] = [training_dict[key][i] for i in shuff_idx]

            # loop over training data
            t = tqdm(range(int(len(training_dict['pid_list']) / model.batch_size) + 1), desc='batch iterations')
            for batch_idxs in batch(range(len(training_dict['pid_list'])), model.batch_size):
                t.update(1)

                #vital_features = [ torch.FloatTensor(training_dict['vital_features_list'][j].values) for j in batch_idxs ]
                #lab_features = [ torch.FloatTensor(training_dict['lab_features_list'][j].values) for j in batch_idxs ]
                #baseline_features = [ torch.FloatTensor(training_dict['baseline_features_list'][j].values) for j in batch_idxs ]
                #labels = [ torch.from_numpy(np.array(training_dict['labels_list'][j])) for j in batch_idxs]
                labels = [ torch.LongTensor(training_dict['labels_list'][j].values.T) for j in batch_idxs]

                #input_data_ = [ [ torch.FloatTensor(training_dict['vital_features_list'][j].values).unsqueeze(0).cuda(), 
                #                  torch.FloatTensor(training_dict['lab_features_list'][j].values).unsqueeze(0).cuda(), 
                #                  torch.FloatTensor(training_dict['baseline_features_list'][j].values).unsqueeze(0).cuda()] for j in batch_idxs ]

                labels_ = move_to_cuda(labels)
                #vital_features_ = move_to_cuda(vital_features)
                #lab_features_ = move_to_cuda(lab_features)
                #baseline_features_ = move_to_cuda(baseline_features)

                input_data_ = [move_to_cuda(training_dict['input_list'][j]) for j in batch_idxs]  # tuple of val ind, val, task ind


                loss = model.estimator.update(input_data_, labels_)
                loss_iteration_k.append(loss)
                loss_history.append(loss)
                
                t.set_description(desc ='loss: {:.4f}'.format(loss), refresh=True)

                #if i > 0 and i % 200 == 0:
                    #output = model.estimator.predict(input_data_)
                    #embeddings = model.estimator.model.embeddings
                    #model.tr_output_list.append(np.array(output.cpu().data))
                    #model.tr_embeddings_list.append(np.array(embeddings.cpu().data))

                del input_data_
                del labels_
            t.set_description(desc ='loss: {:.4f}'.format(np.mean(loss_iteration_k)), refresh=True)
            loss_iterations.append(np.mean(loss_iteration_k))
            if k > 1 and k % 5 == 0:
                tr_util.append(model.evaluate(training_dict,k, True, model))
                val_util.append(model.evaluate(validation_dict,k, False, model))
                tqdm.write('tr_util: ' + str(tr_util))
                tqdm.write('val_util: ' + str(val_util))
            if k > 10 and k % 25 == 0:
                model.save(Path('./model_' + str(k) + '_.out'), model)
                plt.plot(loss_history)
                plt.savefig('loss.png')
                plt.clf()

                plt.plot(tr_util)
                plt.plot(val_util)
                plt.savefig('utility.png')
                plt.clf()
        loss_iterations = pd.DataFrame(loss_iterations, columns = ["iteration"]) 
        return loss_iterations, model

    def postprocess(prediction):
        sep_flag = 0
        pr = []
        for output in prediction:
            if sep_flag == 1:
                pr.append(1)
                continue
            if output[1] > 0.38:
                sep_flag = 1
                pr.append(1)
            else:
                pr.append(0)

        return pr

    @classmethod
    def save(self, model_path, model=None):
        # save torch model
        if model:
            torch.save(model.estimator.state_dict(), "./model.out")
        else:
            torch.save(self.model.estimator.state_dict(), "./model.out")
        
    
    @classmethod
    def load(cls, processing_params, model_params, training_params, model_path):
        pass
    
    @classmethod
    def evaluate(self, testing_dict, j=0, tr=False, model=None):
        print('preprocessing')
        results = []
        testing_dict['input_list'] = []
        for i, _ in enumerate(testing_dict['pid_list']):
            not_nan_indexes = []
            task_indexes = []
            anchor_values = []
            for ii,col in enumerate(testing_dict['vital_features_list'][0].columns.tolist()):
                if col == "index":
                    continue
                iii = ii - 1
                testing_dict['vital_features_list'][i][col] = np.log(testing_dict['vital_features_list'][i][col])
                if testing_dict['vital_features_list'][i][col].count() == 0:
                    testing_dict['vital_features_list'][i][col] = testing_dict['vital_features_list'][i][col].fillna(0)
                testing_dict['vital_features_list'][i][col] = (testing_dict['vital_features_list'][i][col] - testing_dict['vital_features_list'][i][col].mean())/testing_dict['vital_features_list'][i][col].std(ddof=0)
                testing_dict['vital_features_list'][i][col] = testing_dict['vital_features_list'][i][col].fillna(0)

                #elif testing_dict['vital_features_list'][i][col].count() == 1:
                #    testing_dict['vital_features_list'][i][col] = testing_dict['vital_features_list'][i][col].fillna(testing_dict['vital_features_list'][i][col].mean())
                #elif testing_dict['vital_features_list'][i][col].count() <= 3:
                #    testing_dict['vital_features_list'][i][col] = testing_dict['vital_features_list'][i][col].interpolate(method='linear', axis=0).interpolate(method='nearest')
                
                #if testing_dict['vital_features_list'][i][col].count() > 3:
                #    testing_dict['vital_features_list'][i][col] = testing_dict['vital_features_list'][i][col].interpolate(method='pchip', order=3, axis=0).interpolate(method='nearest')
                #testing_dict['vital_features_list'][i][col] = testing_dict['vital_features_list'][i][col].fillna(0.0)

                not_nan = testing_dict['vital_features_list'][i][col].notnull()
                nn_ind = torch.FloatTensor(not_nan[not_nan == True].index.values)
                nn_vals = torch.FloatTensor(testing_dict['vital_features_list'][i][col][not_nan == True].values)
                task_ind = torch.full_like(nn_vals, dtype=torch.long, fill_value=iii)

                not_nan_indexes.append(nn_ind)
                task_indexes.append(task_ind)
                anchor_values.append(nn_vals)

            full_train_x = torch.cat(not_nan_indexes)
            full_train_i = torch.cat(task_indexes)
            full_train_y = torch.cat(anchor_values)

            testing_dict['input_list'].append([full_train_x.cuda(), full_train_i.cuda(), full_train_y.cuda()])

        # Compute utility.
        dt_early   = -12
        dt_optimal = -6
        dt_late    = 3

        max_u_tp = 1
        min_u_fn = -2
        u_fp     = -0.05
        u_tn     = 0
        num_patients = len(testing_dict['pid_list'])
        observed_utilities = np.zeros(num_patients)
        best_utilities     = np.zeros(num_patients)
        worst_utilities    = np.zeros(num_patients)
        inaction_utilities = np.zeros(num_patients)

        for i, pid in enumerate(testing_dict['pid_list']):
            #vital_features = torch.FloatTensor(testing_dict['vital_features_list'][i].values)
            #lab_features = torch.FloatTensor(testing_dict['lab_features_list'][i].values)
            #baseline_features = torch.FloatTensor(testing_dict['baseline_features_list'][i].values)
            labels = torch.LongTensor(testing_dict['labels_list'][i].values.T)
            input_data_ = testing_dict['input_list'][i]
            
            #input_data = [vital_features.unsqueeze(0), lab_features, baseline_features]

            #input_data_ = move_to_cuda(input_data)
            labels_ = labels.cuda()
            seq_length = labels.shape[1]
            if model:
                output = model.estimator.predict(input_data_, seq_length) 
            else:
                output = self.model.estimator.predict(input_data_, seq_length)   
            output = output.tolist()   
            predicted_label = self.postprocess(output) 

            # utility
            labels = np.ravel(testing_dict['labels_list'][i])
            num_rows          = len(labels)
            observed_predictions = predicted_label
            best_predictions     = np.zeros(num_rows)
            worst_predictions    = np.zeros(num_rows)
            inaction_predictions = np.zeros(num_rows)

            if np.any(labels):
                t_sepsis = np.argmax(labels) - dt_optimal
                best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, num_rows)] = 1
            worst_predictions = 1 - best_predictions

            observed_utilities[i] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
            best_utilities[i]     = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
            worst_utilities[i]    = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
            inaction_utilities[i] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

            results.append(pd.DataFrame(list(zip([pid]*len(output), [o[0] for o in output], [o[1] for o in output], predicted_label, labels)), columns = ["pid","p0", "p1", "prediction", "label"]))

        unnormalized_observed_utility = np.sum(observed_utilities)
        unnormalized_best_utility     = np.sum(best_utilities)
        unnormalized_worst_utility    = np.sum(worst_utilities)
        unnormalized_inaction_utility = np.sum(inaction_utilities)
        normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)

        if tr == False:
            eval_results = pd.concat(results)
            eval_results.to_csv('./results_'+str(j)+'_.csv')
        return normalized_observed_utility

    @classmethod
    def exp(self, dataset_load_path='./dataset_balanced6.gpickle'):
        """
        trains model
        :param dataset_load_path: (optional) load path for training/test/validation data. if not specified, loads from experiment_path
        :return: None
        """
        self.dataset = Dataset.load(dataset_load_path)
        
        self.loss_iterations, self.model = self.train(self.dataset.groups['train'], self.dataset.groups['test'])

        self.model.evaluate(self.dataset.groups['test'])
        self.model.save(Path('./model.out'))
        self.loss_iterations.to_csv("./loss_iterations.csv")

        #plt.plot(self.loss_iterations["iteration"].tolist())
        #plt.savefig('loss.png')
        #self.eval("training_model_results", self.dataset.groups['train'])
        #self.eval("validation_model_results", self.dataset.groups['val'])
        #self.eval("testing_model_results", self.dataset.groups['test'])

if __name__ == '__main__':
    fire.Fire(Experiment)