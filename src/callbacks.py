from catalyst.dl.core import Callback, RunnerState
from catalyst.dl.callbacks import CriterionCallback
from catalyst.dl.utils.criterion import accuracy
import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging
from slack_logger import SlackHandler, SlackFormatter


class SlackLogger(Callback):
    """
    Logger callback, translates state.metrics to console and text file
    """

    def __init__(self, url, channel):
        self.logger = None
        self.url = url
        self.channel = channel

    @staticmethod
    def _get_logger(url, channel):
        logger = logging.getLogger("metrics")
        logger.setLevel(logging.INFO)

        slackhandler = SlackHandler(
            username='logger',
            icon_emoji=':robot_face:',
            url=url,
            channel=channel
        )
        slackhandler.setLevel(logging.INFO)

        formater = SlackFormatter()
        slackhandler.setFormatter(formater)
        logger.addHandler(slackhandler)

        return logger

    def on_stage_start(self, state: RunnerState):
        self.logger = self._get_logger(self.url, self.channel)

    def on_stage_end(self, state):
        self.logger.handlers = []

    def on_epoch_end(self, state):
        pass
        # import pdb
        # pdb.set_trace()
        # self.logger.info("", extra={"state": state})


class LabelSmoothCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        loss = criterion(
            state.output[self.output_key],
            state.input[self.input_key]
        )
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class SmoothMixupCallback(LabelSmoothCriterionCallback):
    """
    Callback to do mixup augmentation.
    Paper: https://arxiv.org/abs/1710.09412
    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
        self,
        fields: List[str] = ("images",),
        alpha=0.5,
        on_train_only=True,
        **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        return loss


class DSCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
        loss_weights: List[float] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.loss_weights = loss_weights

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs = state.output[self.output_key]
        input = state.input[self.input_key]
        assert len(self.loss_weights) == len(outputs)
        loss = 0
        for i, output in enumerate(outputs):
            loss += criterion(output, input) * self.loss_weights[i]
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class DSMixupCallback(DSCriterionCallback):
    """
    Callback to do mixup augmentation.

    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """

    def __init__(
        self,
        fields: List[str] = ("features",),
        alpha=1.0,
        on_train_only=True,
        **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        outputs = state.output[self.output_key]
        input_a = state.input[self.input_key]
        input_b = state.input[self.input_key][self.index]
        assert len(self.loss_weights) == len(outputs)
        loss = 0
        for i, output in enumerate(outputs):
            loss_ = self.lam * criterion(output, input_a) + \
                   (1 - self.lam) * criterion(output, input_b)
            loss += loss_ * self.loss_weights[i]

        return loss


class TwoHeadsCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
        loss_weights: List[float] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.loss_weights = loss_weights

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs = state.output[self.output_key]
        outputs1 = state.output["logits1"]
        input_sirna = state.input[self.input_key]
        input_cell = state.input['cell_type']
        loss = 0

        loss += criterion(outputs, input_sirna)
        loss += nn.CrossEntropyLoss()(outputs1, input_cell)

        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class DSAccuracyCallback(Callback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "acc",
        logit_names: List[str] = None,
    ):
        self.prefix = prefix
        self.metric_fn = accuracy
        self.input_key = input_key
        self.output_key = output_key
        self.logit_names = logit_names

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        assert len(outputs) == len(self.logit_names)

        batch_metrics = {}

        for logit_name, output in zip(self.logit_names, outputs):
            metric = self.metric_fn(output, targets)
            key = f"{self.prefix}_{logit_name}"
            batch_metrics[key] = metric[0]

        state.metrics.add_batch_value(metrics_dict=batch_metrics)
