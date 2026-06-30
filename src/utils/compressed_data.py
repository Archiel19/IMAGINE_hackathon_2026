from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from compress_imagenet import (
    compressed_datasets_exist,
    compress_splits,
    jpeg_only_dir_name,
    resized_dir_name,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def apply_compressed_data_config(cfg: DictConfig) -> None:
    """Derive datamodule directories and two-phase settings from ``cfg.data``."""
    data_cfg = cfg.data
    resize_size = data_cfg.resize_size
    crop_size = data_cfg.crop_size
    jpeg_quality = data_cfg.jpeg_quality

    phase1_train = resized_dir_name("train", resize_size, crop_size, jpeg_quality)
    phase1_val = resized_dir_name("val", resize_size, crop_size, jpeg_quality)
    phase2_train = jpeg_only_dir_name("train", jpeg_quality)
    phase2_val = jpeg_only_dir_name("val", jpeg_quality)

    with open_dict(cfg.datamodule):
        cfg.datamodule.train_dir = phase1_train
        cfg.datamodule.val_dir = phase1_val
        cfg.datamodule.phase2_train_dir = phase2_train
        cfg.datamodule.phase2_val_dir = phase2_val
        cfg.datamodule.switch_epoch = data_cfg.switch_epoch

    log.info(
        "Compressed data config: phase 1 train=%s, val=%s; phase 2 train=%s, val=%s; "
        "switch_epoch=%s",
        phase1_train,
        phase1_val,
        phase2_train,
        phase2_val,
        data_cfg.switch_epoch,
    )


def prepare_compressed_datasets(cfg: DictConfig) -> None:
    """Check for existing compressed datasets and run compression when needed."""
    data_cfg = cfg.data
    data_dir = Path(cfg.datamodule.data_path)
    resize_size = data_cfg.resize_size
    crop_size = data_cfg.crop_size
    jpeg_quality = data_cfg.jpeg_quality
    overwrite = data_cfg.get("overwrite", False)

    resized_train = resized_dir_name("train", resize_size, crop_size, jpeg_quality)
    resized_val = resized_dir_name("val", resize_size, crop_size, jpeg_quality)
    jpeg_train = jpeg_only_dir_name("train", jpeg_quality)
    jpeg_val = jpeg_only_dir_name("val", jpeg_quality)

    log.info(
        "Checking compressed datasets (resize=%s, crop=%s, quality=%s, overwrite=%s): "
        "resized=[%s, %s], jpeg-only=[%s, %s]",
        resize_size,
        crop_size,
        jpeg_quality,
        overwrite,
        resized_train,
        resized_val,
        jpeg_train,
        jpeg_val,
    )

    if not overwrite and compressed_datasets_exist(
        data_dir, resize_size, crop_size, jpeg_quality
    ):
        log.info(
            "All compressed datasets already exist under %s and overwrite=false; "
            "skipping compression.",
            data_dir,
        )
        print(
            f"[compressed_data] Skipping compression: resized and JPEG-only datasets "
            f"already exist in {data_dir} (set data.overwrite=true to reprocess)."
        )
        return

    if overwrite:
        log.info("overwrite=true: reprocessing all compressed datasets.")
        print("[compressed_data] overwrite=true: running compression (reprocessing all images).")
    else:
        log.info("One or more compressed datasets missing; running compression.")
        print("[compressed_data] Running compression to create missing datasets.")

    compress_splits(
        data_dir=data_dir,
        resize_size=resize_size,
        crop_size=crop_size,
        jpeg_quality=jpeg_quality,
        also_jpeg_only=True,
        overwrite=overwrite,
    )

    log.info("Compressed dataset preparation finished.")
    print("[compressed_data] Compression finished.")
