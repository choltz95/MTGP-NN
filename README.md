# MTGP-NN
Early sepsis detection via multitask gaussian process cnn-rnn 

We follow the general framework of [1] with several extensions. A Multitask Gaussian Process [2] is employed to normalize the time-scale of irregularly sampled clinical data (vital signs). Any prediction algorithm can be employed on top of this framework, but we process this data with a cnn-based encoder and perform temporal classification with a GRU-LSTM augmented with auxilliary features (patient demographics & coursened lab data statistics). Multitask learning is induced via prediction of hospital stay durration. An implementation of gradnorm [3] is included to learn loss-weights and the whole framework is optimized end-to-end with backprop.

GPytorch [3] is used for MTGP interpolation & Pytorch [4] is used for the neural network.

Results to be included.

[1] Joseph Futoma, Sanjay Hariharan, Katherine Heller, Learning to Detect Sepsis with a Multitask Gaussian Process RNN Classifier, ICML'17 - https://arxiv.org/abs/1706.04152
[2] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich, GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks, ICML 2018 - https://arxiv.org/abs/1711.02257
[3] https://gpytorch.ai/
[4] https://arxiv.org/abs/1711.02257
