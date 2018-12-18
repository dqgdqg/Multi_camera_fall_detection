from .builder import *
from ..utils import load_model_weights
from ..weights import weights_collection


def ResNet18_bottom(input_id, input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    input_shape = input_tensor.shape
    y = build_bottom_resnet(input_id=input_id,
                             input_tensor=input_tensor,
                             input_shape=input_shape,
                             repetitions=(2, 2, 2, 2),
                             classes=classes,
                             include_top=include_top,
                             block_type='basic')
    return y

def ResNet18_top(x, weights=None, classes=2, include_top=False):
    y = build_top_resnet(x=x,
                             repetitions=(2, 2, 2, 2),
                             classes=classes,
                             include_top=include_top,
                             block_type='basic')
    
    return y