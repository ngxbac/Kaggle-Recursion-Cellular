from typing import Mapping, Any
from catalyst.dl.core import Runner


class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        # import pdb
        # pdb.set_trace()
        if 'group_labels' in batch:
            output = self.model(batch["images"], batch['group_labels'])
        else:
            output = self.model(batch["images"])
        return {
            "logits": output
        }
