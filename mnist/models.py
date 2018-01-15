import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):
    def __init__(self, n_out):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, pad=1)
            self.conv2 = L.Convolution2D(None, 32, 3, pad=1)
            self.conv3 = L.Convolution2D(None, 64, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 64, 3, pad=1)
            self.fc1 = L.Linear(None, 128)
            self.fc2 = L.Linear(None, n_out)

    def __call__(self, x):
        h = F.reshape(x, (-1, 1, 28, 28))
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, 0.25)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, 0.25)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, 0.5)
        h = F.relu(self.fc2(h))
        return h


class CNN_keras(chainer.Chain):
    def __init__(self, n_out):
        super(CNN_keras, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, pad=1)
            self.conv2 = L.Convolution2D(None, 64, 3, pad=1)
            self.fc1 = L.Linear(None, 128)
            self.fc2 = L.Linear(None, n_out)

    def __call__(self, x):
        h = F.reshape(x, (-1, 1, 28, 28))
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, 0.25)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, 0.5)
        h = F.relu(self.fc2(h))
        return h


class CNN2(chainer.Chain):
    def __init__(self, n_out):
        super(CNN2, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, pad=2)
            self.conv2 = L.Convolution2D(None, 64, 3, pad=2)
            self.conv3 = L.Convolution2D(None, 128, 3, pad=2)
            self.fc4 = L.Linear(None, 512)
            self.fc5 = L.Linear(None, n_out)

    def __call__(self, x):
        h = F.reshape(x, (-1, 1, 28, 28))
        h = F.relu(self.conv1(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.fc4(h)))
        h = F.relu(self.fc5(h))
        return h


# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out, dropout=False):
        super(MLP, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        if self.dropout:
            h1 = F.dropout(F.relu(self.l1(x)))
            h2 = F.dropout(F.relu(self.l2(h1)))
        else:
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))
        return self.l3(h2)
