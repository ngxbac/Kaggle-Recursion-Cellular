# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *


# Register models
registry.Model(cell_resnet)
registry.Model(cell_senet)
registry.Model(cell_densenet)
registry.Model(EfficientNet)

# Register callbacks
registry.Callback(LabelSmoothCriterionCallback)
registry.Callback(SmoothMixupCallback)

# Register criterions
registry.Criterion(LabelSmoothingCrossEntropy)

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)