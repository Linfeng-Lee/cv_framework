import abc
from typing import Callable


class BaseTrainer(abc.ABC):

    @abc.abstractmethod
    def create_model(self, model_names: str, pretrained: bool, **kwargs):
        ...

    @abc.abstractmethod
    def resume(self, model_path: str, strict: bool = False, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def save_checkpoint(self, state, save_path: str, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def define_scalar(self, save_path: str, comment: str, **kwargs):
        ...

    @abc.abstractmethod
    def define_optimizer(self, lr: float, **kwargs):
        ...

    @abc.abstractmethod
    def define_loader(self, path: str, **kwargs):
        ...

    @abc.abstractmethod
    def define_criterion(self, criterion_list: list, gpus: int, **kwargs):
        ...

    @abc.abstractmethod
    def define_lr_scheduler(self, optimizer: Callable, **kwargs):
        ...

    @abc.abstractmethod
    def fit(self, save_path: str, **kwargs):
        ...

    @abc.abstractmethod
    def train_epoch(self, epoch: int, **kwargs):
        ...

    @abc.abstractmethod
    def validate(self, epoch: int, **kwargs):
        ...
