# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *
from schedulers import *


# Register models
registry.Model(ResNet)
registry.Model(cell_senet)
registry.Model(cell_densenet)
registry.Model(SENetGrouplevel)
registry.Model(EfficientNet)
registry.Model(SENetTIMM)
registry.Model(InceptionV3TIMM)
registry.Model(GluonResnetTIMM)
registry.Model(DSInceptionV3)
registry.Model(DSSENet)
registry.Model(ResNet50CutMix)
registry.Model(Fishnet)
registry.Model(SENetCellType)
registry.Model(SENetCellMultipleDropout)
registry.Model(MixNet)

# Register callbacks
registry.Callback(LabelSmoothCriterionCallback)
registry.Callback(SmoothMixupCallback)
registry.Callback(DSAccuracyCallback)
registry.Callback(DSCriterionCallback)
registry.Callback(SlackLogger)
registry.Callback(TwoHeadsCriterionCallback)

# Register criterions
registry.Criterion(LabelSmoothingCrossEntropy)

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)
registry.Optimizer(RAdam)

registry.Scheduler(CyclicLRFix)