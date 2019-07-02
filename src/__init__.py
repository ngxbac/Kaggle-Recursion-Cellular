# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *

registry.Model(cell_resnet)
registry.Model(cell_senet)
registry.Model(cell_densenet)