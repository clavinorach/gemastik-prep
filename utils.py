import os, random, json
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n; self.cnt += n
    @property
    def avg(self): return self.sum / max(1, self.cnt)

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f: return json.load(f)
