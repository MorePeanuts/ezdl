import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def show_images(imgs, n_rows, n_cols, titles=None, scale=1.5):
    figsize = (n_cols * scale, n_rows * scale)
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() # type: ignore
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.detach().numpy()
        except Exception:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class SyntheticRegressionData(Dataset):
    
    def __init__(self, w, b, noise=0.01, total_samples=2000, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.total_samples = total_samples
        
        if isinstance(w, torch.Tensor):
            self.w = w.reshape(-1, 1)
        else:
            self.w = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)
            
        if not isinstance(b, torch.Tensor):
            self.b = torch.tensor(b, dtype=torch.float32)
        else:
            self.b = b
        
        self.x = torch.randn(total_samples, len(w))
        noise = torch.randn(total_samples, 1) * noise
        self.y = self.x @ self.w + self.b + noise
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        