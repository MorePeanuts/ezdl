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


def format_instruction(
    entry: dict,
    instruction_template: str | None = None, 
    response_template: str | None = None,
    instruction_key='instruction',
    input_key='input',
    output_key='output'
):
    if instruction_template is None:
        instruction_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\n{instruction}"
            "\n\n### Input:\n{input}"
        )
    if response_template is None:
        response_template = "\n\n### Response:\n{output}"
        
    instruction_text = instruction_template.format(
        instruction=entry.get(instruction_key, ''),
        input=entry.get(input_key, ''),
    )
    response_text = response_template.format(
        output=entry.get(output_key, '')
    )
    
    return instruction_text, response_text


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
        