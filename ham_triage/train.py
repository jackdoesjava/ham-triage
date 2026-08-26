import argparse
import json
import math
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

from .config import CLASSES, Paths, TrainConfig
from .data import load_cache

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def to_input(x: torch.Tensor, crop: int, train: bool) -> torch.Tensor:
    # x is uint8 [B, 256, 256, 3] already on the GPU. Augmenting here rather than in
    # DataLoader workers avoids Windows spawning workers that each pickle a 2 GB array.
    b, side = x.shape[0], x.shape[1]
    dev = x.device
    if train:
        oy = torch.randint(0, side - crop + 1, (b,), device=dev)
        ox = torch.randint(0, side - crop + 1, (b,), device=dev)
    else:
        oy = ox = torch.full((b,), (side - crop) // 2, device=dev)
    ar = torch.arange(crop, device=dev)
    rows = (oy[:, None] + ar)[:, :, None]
    cols = (ox[:, None] + ar)[:, None, :]
    x = x[torch.arange(b, device=dev)[:, None, None], rows, cols]
    x = x.permute(0, 3, 1, 2).float()
    if train:
        # two flips and a transpose generate the whole dihedral group; dermoscopy has no up
        flip = torch.rand(3, b, 1, 1, 1, device=dev) < 0.5
        x = torch.where(flip[0], x.flip(2), x)
        x = torch.where(flip[1], x.flip(3), x)
        x = torch.where(flip[2], x.transpose(2, 3), x)
        contrast = 1 + 0.2 * (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
        brightness = 25 * (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
        x = (x - 128) * contrast + 128 + brightness
    return (x - MEAN.to(dev)) / STD.to(dev)


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    return timm.create_model(cfg.model, pretrained=True, num_classes=len(CLASSES),
                             drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate)


def train(cfg: TrainConfig, images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
          seed: int, device: str = "cuda"):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_idx) // cfg.batch_size
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    rng = np.random.default_rng(seed)
    curve = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train_idx)
        loss_sum = 0.0
        for i in range(steps_per_epoch):
            idx = np.sort(order[i * cfg.batch_size:(i + 1) * cfg.batch_size])
            x = torch.from_numpy(images[idx]).to(device, non_blocking=True)
            y = torch.from_numpy(labels[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                logits = model(to_input(x, cfg.crop, train=True))
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
        curve.append(loss_sum / steps_per_epoch)
        print(f"epoch {epoch + 1}/{cfg.epochs}  loss {curve[-1]:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
    return model, curve


@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, cfg: TrainConfig,
            device: str = "cuda", batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device)
        # fp32 on purpose: fp16 logits produce ties in the conformal scores later on
        out.append(model(to_input(x, cfg.crop, train=False)).float().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, help="name of a parquet in results/splits")
    p.add_argument("--train-col", default="train", help="boolean column giving the training set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", required=True)
    p.add_argument("--loss", default="ce")
    p.add_argument("--epochs", type=int)
    args = p.parse_args()
    cfg = TrainConfig(loss=args.loss)
    if args.epochs:
        cfg = replace(cfg, epochs=args.epochs)

    paths = Paths()
    images, meta = load_cache(paths)
    split = pd.read_parquet(paths.results / "splits" / f"{args.split}.parquet")
    assert (split.image_id.values == meta.image_id.values).all()
    train_idx = np.flatnonzero(split[args.train_col].values)
    labels = meta.label.values.astype(np.int64)

    t0 = time.time()
    model, curve = train(cfg, images, labels, train_idx, args.seed)
    logits = predict(model, images, cfg)

    run_dir = paths.results / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(logits, columns=[f"logit_{c}" for c in CLASSES])
    frame.insert(0, "image_id", meta.image_id.values)
    frame.to_parquet(run_dir / "logits.parquet", index=False)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    minutes = round((time.time() - t0) / 60, 1)
    info = {
        "run_id": args.run_id, "seed": args.seed, "split": args.split, "train_col": args.train_col,
        "n_train": int(len(train_idx)), "config": asdict(cfg), "classes": list(CLASSES),
        "train_loss": curve, "minutes": minutes,
        "torch": torch.__version__, "timm": timm.__version__,
        "pretrained": model.pretrained_cfg.get("hf_hub_id"), "device": torch.cuda.get_device_name(0),
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {run_dir}  ({minutes} min)")


if __name__ == "__main__":
    main()
