# infer_cifar10.py
import os, argparse, json
import torch
from PIL import Image
from torchvision import transforms
import timm

def load_labels(path):
    with open(path) as f: return json.load(f)

def build_tf(img_size=224):
    mean, std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

@torch.no_grad()
def predict_image(model, tf, img_path, device, classes):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(prob, dim=0)
    return classes[idx.item()], conf.item()

def load_model(ckpt_path, labels_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = load_labels(labels_path)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "vit_tiny_patch16_224")
    num_classes = len(classes)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, classes, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/best_vit_cifar10.pth")
    ap.add_argument("--labels", type=str, default="outputs/labels.json")
    ap.add_argument("--img", type=str, help="path gambar tunggal (opsional)")
    ap.add_argument("--folder", type=str, help="folder berisi gambar untuk diprediksi (opsional)")
    ap.add_argument("--out", type=str, default="predictions.csv")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, classes, cfg = load_model(args.ckpt, args.labels, device)
    tf = build_tf(img_size=cfg.get("img_size", 224))

    if args.img:
        label, conf = predict_image(model, tf, args.img, device, classes)
        print(f"{os.path.basename(args.img)},{label},{conf:.4f}")
        return

    if args.folder:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename","pred","confidence"])
            for fn in os.listdir(args.folder):
                p = os.path.join(args.folder, fn)
                if not os.path.isfile(p): continue
                try:
                    label, conf = predict_image(model, tf, p, device, classes)
                    w.writerow([fn, label, f"{conf:.6f}"])
                except Exception as e:
                    w.writerow([fn, "ERROR", str(e)])
        print("Tersimpan:", args.out)
        return

    print("Gunakan --img atau --folder untuk melakukan inferensi.")

if __name__ == "__main__":
    main()
