# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *


registry.Model(cell_resnet)
registry.Model(cell_senet)
registry.Model(cell_densenet)

registry.Callback(LabelSmoothCriterionCallback)

registry.Criterion(LabelSmoothingCrossEntropy)