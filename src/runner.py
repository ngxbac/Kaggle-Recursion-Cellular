from typing import Mapping, Any
from catalyst.dl.core import Runner
from models import GAIN
import torch


class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if isinstance(model, GAIN):
            output, output_am = self.model(batch["images"], batch['targets'])
            return {
                "logits": output,
                "logits_am": output_am,
            }
        else:
            if 'group_labels' in batch:
                output = self.model(batch["images"], batch['group_labels'])
            else:
                output = self.model(batch["images"])
            return {
                "logits": output
            }
