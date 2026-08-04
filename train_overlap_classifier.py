import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = PROJECT_DIR.parent / "data_esrf" / "overlap_classifier_train.h5"
DEFAULT_VAL_FILE = PROJECT_DIR.parent / "data_esrf" / "overlap_classifier_validation.h5"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "overlap_classifier"


class H5OverlapClassifierDataset(Dataset):
    def __init__(self, h5_file: Path, img_scale: float = 1.0):
        self.h5_file = Path(h5_file)
        self.img_scale = float(img_scale)
        if not self.h5_file.exists():
            raise FileNotFoundError(f"HDF5 classifier file not found: {self.h5_file}")

        with h5py.File(self.h5_file, "r") as f:
            for name in ("images", "labels"):
                if name not in f:
                    raise ValueError(f"{self.h5_file} does not contain dataset {name!r}")
            self.length = int(f["labels"].shape[0])
            self.image_shape = tuple(f["images"].shape[1:])
            self.label_counts = np.bincount(f["labels"][:].astype(np.int64), minlength=2)

    def __len__(self):
        return self.length

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        finite = np.isfinite(image)
        if not finite.any():
            return np.zeros_like(image, dtype=np.float32)

        values = image[finite]
        lo, hi = np.percentile(values, [1, 99.9])
        if hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)

        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
        image[~finite] = 0.0
        return image.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            image = f["images"][idx]
            label = float(f["labels"][idx])

        image = self.normalize_image(image)
        image = torch.from_numpy(image).unsqueeze(0)

        if self.img_scale != 1.0:
            new_size = (
                max(1, int(round(image.shape[-2] * self.img_scale))),
                max(1, int(round(image.shape[-1] * self.img_scale))),
            )
            image = F.interpolate(
                image.unsqueeze(0),
                size=new_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FourLayerOverlapCNN(nn.Module):
    def __init__(self, base_channels: int = 16, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 8),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= 0.5
    labels_bool = labels >= 0.5

    true_positive = (predictions & labels_bool).sum().item()
    false_positive = (predictions & ~labels_bool).sum().item()
    false_negative = (~predictions & labels_bool).sum().item()
    true_negative = (~predictions & ~labels_bool).sum().item()
    total = max(1, labels.numel())

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": predictions.float().mean().item(),
    }


def evaluate(model, dataloader, criterion, device, amp: bool) -> dict[str, float]:
    model.eval()
    losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.float32)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(loss.detach())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    model.train()
    if not losses:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "positive_rate": 0.0}

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = binary_metrics_from_logits(logits, labels)
    metrics["loss"] = torch.stack(losses).mean().item()
    return metrics


def limit_dataset(dataset: Dataset, max_samples: int | None, seed: int = 0) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:max_samples]
    return Subset(dataset, indices.tolist())


def make_run_name(train_file: Path, epochs: int, batch_size: int, lr: float, img_scale: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{train_file.stem}_e{epochs}_b{batch_size}_lr{lr:g}_s{img_scale:g}"


def train(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    train_dataset_full = H5OverlapClassifierDataset(args.train_file, img_scale=args.scale)
    val_dataset_full = H5OverlapClassifierDataset(args.val_file, img_scale=args.scale)
    train_dataset = limit_dataset(train_dataset_full, args.max_samples, seed=0)
    val_dataset = limit_dataset(val_dataset_full, args.max_val_samples, seed=1)

    train_counts = train_dataset_full.label_counts.astype(np.float64)
    pos_weight_value = train_counts[0] / max(1.0, train_counts[1])
    if args.pos_weight is not None:
        pos_weight_value = args.pos_weight

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type != "cpu",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    model = FourLayerOverlapCNN(base_channels=args.base_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    run_name = args.run_name or make_run_name(args.train_file, args.epochs, args.batch_size, args.learning_rate, args.scale)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "runs" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    logging.info("Train file: %s", args.train_file)
    logging.info("Validation file: %s", args.val_file)
    logging.info("Train samples: %d", len(train_dataset))
    logging.info("Validation samples: %d", len(val_dataset))
    logging.info("Train label counts from full file: single=%d overlap=%d", int(train_counts[0]), int(train_counts[1]))
    logging.info("BCE positive weight: %.3f", pos_weight_value)
    logging.info("TensorBoard: tensorboard --logdir %s", output_dir / "runs")

    writer.add_hparams(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "scale": args.scale,
            "base_channels": args.base_channels,
            "dropout": args.dropout,
            "pos_weight": pos_weight_value,
            "train_file": str(args.train_file),
            "val_file": str(args.val_file),
        },
        {"hparam/metric": 0.0},
    )

    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in progress:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            loss_value = loss.detach().item()
            epoch_loss += loss_value
            progress.set_postfix(loss=loss_value)
            writer.add_scalar("Loss/train_batch", loss_value, global_step)

        train_loss = epoch_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion, device, args.amp)
        scheduler.step(val_metrics["loss"])

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("Metrics/precision", val_metrics["precision"], epoch)
        writer.add_scalar("Metrics/recall", val_metrics["recall"], epoch)
        writer.add_scalar("Metrics/f1", val_metrics["f1"], epoch)
        writer.add_scalar("Metrics/predicted_positive_rate", val_metrics["positive_rate"], epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        logging.info(
            "Epoch %d: train_loss=%.5f val_loss=%.5f acc=%.4f precision=%.4f recall=%.4f f1=%.4f pred_pos=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["positive_rate"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_dir / f"{run_name}_best.pth",
            )
            logging.info("Saved new best checkpoint from epoch %d", epoch)

        if args.save_every and epoch % args.save_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"{run_name}_epoch{epoch}.pth")

    torch.save(model.state_dict(), checkpoint_dir / f"{run_name}_final.pth")
    writer.close()
    logging.info("Finished. Best validation loss %.5f at epoch %d", best_val_loss, best_epoch)


def get_args():
    parser = argparse.ArgumentParser(description="Train a small four-block CNN to classify single spots vs synthetic overlaps.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scale", "-s", type=float, default=0.5, help="Downscale images before the CNN to reduce memory use.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision when supported.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 0))
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=None, help="Override BCE positive class weight. Default: negatives / positives.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit train samples for a quick debug run.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples for a quick debug run.")
    parser.add_argument("--save-every", type=int, default=0, help="Also save plain state_dict checkpoints every N epochs.")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
