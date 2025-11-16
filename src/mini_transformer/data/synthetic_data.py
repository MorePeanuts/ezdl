import torch
from torch.utils.data import Dataset
from ..plot_utils import plot_data_points


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


class SyntheticSineData(Dataset):
    def __init__(self, total_steps=1000, window_size=4):
        self.time = torch.arange(1, total_steps + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(total_steps) * 0.2
        self.window_size = window_size
        self.total_steps = total_steps
        self.features = torch.stack(
            [self.x[i : self.total_steps - self.window_size + i] for i in range(self.window_size)],
            1,
        )
        self.labels = self.x[self.window_size :].reshape((-1, 1))

    def __len__(self):
        return self.total_steps - self.window_size

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def show_dataset(self):
        plot_data_points(
            self.time,
            self.x,
            xlabel='time',
            ylabel='x',
            xlim=[1, 1000],
            figsize=(12, 6),
        )
