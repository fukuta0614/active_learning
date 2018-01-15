from .originals import *
from .cnn_info import CNN_ARCH_INFO
from .bilinear_cnn import BilinearCNN
from .trainable_cnn import TrainableCNN
from .regularized_classifier import TextureRegularizedClassifier, CBPRegularizedClassifier


import os
server = os.uname()[1]

if 'usropsai05' in server:
    MODEL_PATH = '/home/users/usropsai05/.chainer/dataset/pfnet/chainer/models/'
elif server == 'dl-box-docker':
    MODEL_PATH = '/data/chainer_model'
elif server == 'kali-docker':
    MODEL_PATH = '/data1/chainer_model'
else:
    MODEL_PATH = '/data/unagi0/fukuta/chainer_model'
