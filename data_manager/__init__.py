from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .veri776 import *
from .vehicleid import *


__imgreid_factory = {
    'veri776': VeRi776,
    'veri776wgl': VeRi776WithGroupLabel,
    'veri776wcl': VeRi776WithColorLBP,
    'veri776plt': VeRi776Plt,
    'vehicleid': VehicleID,
}


def get_names():
    return list(__imgreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)

