from lightning import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _invalidate_dataloaders(trainer: Trainer) -> None:
    """Teardown active loaders so the next epoch creates fresh ones.

    Lightning calls ``setup_data()`` before ``on_train_epoch_start``, so switching
    datasets inside ``on_train_epoch_start`` and calling ``setup_data()`` again leaves
    orphaned worker processes from the first loader and can hang or crash silently.
    """
    fit_loop = trainer.fit_loop
    if fit_loop._data_fetcher is not None:
        fit_loop._data_fetcher.teardown()
        fit_loop._data_fetcher = None
    fit_loop._combined_loader = None

    val_loop = fit_loop.epoch_loop.val_loop
    if val_loop._data_fetcher is not None:
        val_loop._data_fetcher.teardown()
        val_loop._data_fetcher = None
    val_loop._combined_loader = None


class CompressedDataPhaseSwitchCallback(Callback):
    """Switch from resized+compressed data to JPEG-only data at a configured epoch."""

    def __init__(self, switch_epoch: int) -> None:
        if switch_epoch < 0:
            raise ValueError(f"switch_epoch must be non-negative, got {switch_epoch}")
        self.switch_epoch = switch_epoch

    def _switch_to_phase_2(self, trainer: Trainer, completed_epoch: int) -> None:
        datamodule = trainer.datamodule
        if datamodule is None or getattr(datamodule.hparams, "phase2_train_dir", None) is None:
            return
        if datamodule.current_phase >= 2:
            return

        old_train = datamodule.hparams.train_dir
        old_val = datamodule.hparams.val_dir
        datamodule.switch_to_phase_2()
        _invalidate_dataloaders(trainer)
        self._log_switch(completed_epoch, old_train, old_val, datamodule)

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        if trainer.current_epoch >= self.switch_epoch:
            self._switch_to_phase_2(trainer, trainer.current_epoch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        if trainer.current_epoch + 1 == self.switch_epoch:
            self._switch_to_phase_2(trainer, trainer.current_epoch)

    @rank_zero_only
    def _log_switch(
        self,
        completed_epoch: int,
        old_train: str,
        old_val: str,
        datamodule,
    ) -> None:
        msg = (
            f"Switching to phase 2 (JPEG-only) after epoch {completed_epoch}: "
            f"train {old_train} -> {datamodule.hparams.train_dir}, "
            f"val {old_val} -> {datamodule.hparams.val_dir} "
            f"(online resize/crop enabled from epoch {completed_epoch + 1})"
        )
        log.info(msg)
        print(f"[compressed_data] {msg}")
