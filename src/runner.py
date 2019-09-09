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

    # def _run_stage(self, stage: str):
    #     self._prepare_state(stage)
    #     loaders = self.experiment.get_loaders(stage)
    #     self.callbacks = self.experiment.get_callbacks(stage)
    #
    #     self._run_event("stage_start")
    #     for epoch in range(18, self.state.num_epochs):
    #         self.state.stage_epoch = epoch
    #
    #         self._run_event("epoch_start")
    #         self._run_epoch(loaders)
    #         self._run_event("epoch_end")
    #
    #         if self._check_run and self.state.epoch >= 3:
    #             break
    #         if self.state.early_stop:
    #             self.state.early_stop = False
    #             break
    #
    #         self.state.epoch += 1
    #     self._run_event("stage_end")
