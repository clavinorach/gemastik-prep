# train_optuna_vit.py
import os, time, argparse
from typing import Tuple
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import timm

from utils import set_seed, AverageMeter, save_json

# ------------------------------
# Data pipeline (CIFAR-10)
# ------------------------------
def build_loaders(data_dir: str, img_size: int, batch_size: int, num_workers: int=0) -> Tuple[DataLoader, DataLoader, list]:
    # CIFAR-10 mean/std (ImageNet norm juga OK karena pakai ViT pretrained)
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Offline-friendly: asumsi dataset sudah ada di data_dir
    tr = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    va = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=val_tf)

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(num_workers > 0))
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0))

    return train_loader, val_loader, tr.classes

# ------------------------------
# Train / Validate loops
# ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_meter.update(loss.item(), n=x.size(0))
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / max(1, total)
    return loss_meter.avg, acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_meter.update(loss.item(), n=x.size(0))
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / max(1, total)
    return loss_meter.avg, acc

# ------------------------------
# Optuna Objective
# ------------------------------
def create_model(model_name: str, num_classes: int, pretrained: bool, drop_rate: float):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model

def objective(trial: optuna.Trial, args):
    set_seed(args.seed)

    # Search space - OPTIMIZED untuk GTX 1050 (2GB VRAM)
    model_name = trial.suggest_categorical("model_name", [
        "vit_tiny_patch16_224",     # ~5.7M params - RECOMMENDED untuk GTX 1050
        # "vit_small_patch16_224",  # ~22M params - mungkin OOM
        # "vit_base_patch16_224",   # ~86M params - pasti OOM
    ])
    img_size = trial.suggest_categorical("img_size", [224])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])  # Kurangi dari 32,64 ke 16,32
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
    drop_rate = trial.suggest_float("drop_rate", 0.0, 0.3)

    epochs = args.epochs
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    # Tambahkan print untuk debugging
    print(f"\n{'='*60}")
    print(f"[Trial {trial.number}] Starting...")
    print(f"Model: {model_name}, Batch: {batch_size}, LR: {lr:.2e}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")

    print(f"[Trial {trial.number}] Loading CIFAR-10...")
    train_loader, val_loader, classes = build_loaders(args.data, img_size, batch_size, args.workers)
    num_classes = len(classes)

    print(f"[Trial {trial.number}] Creating model {model_name}...")
    model = create_model(model_name, num_classes, pretrained=not args.no_pretrained, drop_rate=drop_rate).to(device)
    print(f"[Trial {trial.number}] Model loaded. Starting training...\n")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" and args.amp else None

    best_acc = 0.0
    patience = args.patience
    bad = 0

    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        trial.report(va_acc, step=ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # early stop manual sederhana
        if va_acc > best_acc:
            best_acc = va_acc
            bad = 0
            # simpan sementara (per-trial) agar trial terbaik punya checkpoint
            os.makedirs(args.out, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "config": {
                    "model_name": model_name, "img_size": img_size,
                    "lr": lr, "weight_decay": weight_decay, "drop_rate": drop_rate,
                    "epochs": epochs, "batch_size": batch_size
                }
            }, os.path.join(args.out, f"trial_best_tmp.pth"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[Trial {trial.number}] Early stopping at epoch {ep+1}")
                break

        # ALWAYS print progress (GTX 1050 butuh feedback)
        print(f"[Trial {trial.number}] Ep {ep+1:2d}/{epochs} | "
              f"train: {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val: {va_loss:.4f}/{va_acc:.4f} | "
              f"best: {best_acc:.4f}")

    print(f"\n[Trial {trial.number}] Finished with best_acc={best_acc:.4f}\n")
    return best_acc

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data", help="folder CIFAR-10 yang sudah ada")
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=0, help="detik, 0=tanpa batas")
    ap.add_argument("--workers", type=int, default=0, help="num_workers for DataLoader (Windows: use 0)")
    ap.add_argument("--cpu", action="store_true", help="paksa CPU")
    ap.add_argument("--no-pretrained", action="store_true", help="jangan pakai pretrained weights")
    ap.add_argument("--amp", action="store_true", help="mixed precision (GPU)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Simpan label CIFAR-10 untuk inferensi nanti
    # (gunakan loader sekali untuk dapatkan classes)
    try:
        _, _, classes = build_loaders(args.data, 224, 32, num_workers=0)  # FIX: gunakan 0, bukan 2
        save_json(classes, os.path.join(args.out, "labels.json"))
    except Exception as e:
        print("Gagal membuka CIFAR-10. Pastikan sudah tersedia offline di --data :", e)
        return

    # Optuna study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

    print("Mulai HPO...")
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials,
                   timeout=None if args.timeout==0 else args.timeout, gc_after_trial=True)
    
    print("== Ringkasan HPO ==")
    print("Best value (val_acc):", study.best_value)
    print("Best params:", study.best_params)

    # Ganti nama checkpoint terbaik sementara menjadi final
    tmp_ckpt = os.path.join(args.out, "trial_best_tmp.pth")
    best_ckpt = os.path.join(args.out, "best_vit_cifar10.pth")
    if os.path.exists(tmp_ckpt):
        os.replace(tmp_ckpt, best_ckpt)
        print("Model terbaik disimpan ke:", best_ckpt)

    # Simpan ringkasan HPO
    with open(os.path.join(args.out, "best_params.txt"), "w") as f:
        f.write(str(study.best_params) + "\n")
        f.write(f"best_val_acc={study.best_value:.6f}\n")

if __name__ == "__main__":
    main()
