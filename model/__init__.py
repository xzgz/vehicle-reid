from __future__ import absolute_import

from .resnet import *
from .hacnn import *


__model_factory = {
    'resnet50': ResNet50,
    'hacnn': HACNN,
    'mgn': MGN,
    'originmgn': OriginMGN,
    'mgnb4': MGNB4,
    'mgnb2': MGNB2,
    'ressoattn': ResSoAttn,
    'ressohaattn': ResSoHaAttn,
    'resv2soattn': Resv2SoAttn,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)