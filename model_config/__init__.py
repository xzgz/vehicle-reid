from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .veri776_config import VeRi776Config
from .vehicleid_config import VehicleIDConfig


__dataset_config_factory = {
    'veri776': VeRi776Config,
    'vehicleid': VehicleIDConfig,
}


def get_names():
    return list(__dataset_config_factory.keys())


def init_dataset_config(name, **kwargs):
    if name not in list(__dataset_config_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}"
                       .format(name, list(__dataset_config_factory.keys())))
    return __dataset_config_factory[name](**kwargs)

