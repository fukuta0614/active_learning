from chainer.links.model.vision import vgg, resnet, googlenet

CNN_ARCH_INFO = {
    'resnet50': {
        'cnn': resnet.ResNet50Layers,
        'logit': ['fc6'],
        'layer_info': {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048},
        'default_texture_layer': 'res2',
    },
    'resnet152': {
        'cnn': resnet.ResNet152Layers,
        'logit': ['fc6'],
        'layer_info': {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048},
        'default_texture_layer': 'res2',
    },
    'resnet101': {
        'cnn': resnet.ResNet101Layers,
        'logit': ['fc6'],
        'layer_info': {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048},
        'default_texture_layer': 'res2',
    },
    'vgg': {
        'cnn': vgg.VGG16Layers,
        'logit': ['fc8'],
        'layer_info': {'pool3': 256, 'pool4': 512, 'pool5': 512},
        'default_texture_layer': 'pool3',
    },
    'googlenet': {
        'cnn': googlenet.GoogLeNet,
        'logit': ['loss3_fc', 'loss1_fc2', 'loss2_fc2'],
        'default_texture_layer': 'inception_4b',
        'layer_info': {'inception_4b': 512, 'inception_4d': 528, 'pool3': 480, 'pool4': 832}
    }
}
