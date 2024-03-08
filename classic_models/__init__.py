from .alexnet import alexnet
from .vggnet import vgg11, vgg13, vgg16, vgg19
from .googlenet_v1 import googlenet
from .resnet import  resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from .densenet import densenet121, densenet161, densenet169
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from .mobilenet_v1 import mobilenet_version1
from .mobilenet_v2 import mobilenet_version2


cfgs = {
    'alexnet': alexnet,
    'vgg': vgg16,
    'vgg_tiny': vgg11,
    'vgg_small': vgg13,
    'vgg_big': vgg19,
    'googlenet': googlenet,   
    'resnet_small': resnet34,
    'resnet': resnet50,
    'resnet_big': resnet101,
    'resnext': resnext50_32x4d,
    'resnext_big': resnext101_32x8d,
    'densenet_tiny': densenet121,
    'densenet_small': densenet161,
    'densenet': densenet169,
    'densenet_big': densenet121,
    'mobilenet_v1': mobilenet_version1,
    'mobilenet_v2': mobilenet_version2,
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext': convnext_base,
    'convnext_big': convnext_large,
    'convnext_huge': convnext_xlarge,
}

def find_model_using_name(model_name, num_classes):   
    return cfgs[model_name](num_classes)