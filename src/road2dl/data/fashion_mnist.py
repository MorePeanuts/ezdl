import torchvision
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset
from .data_utils import show_images


class FashionMNIST(Dataset):
    
    dataset_path = Path(__file__).parents[3] / 'datasets/fashion_mnist/'
    
    def __init__(self, batch_size=64, resize=(28, 28), train=True):
        super().__init__()
        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        self.data = torchvision.datasets.FashionMNIST(
            root=FashionMNIST.dataset_path,
            train=train,
            transform=trans,
            download=True
        ) 
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
        
    @staticmethod
    def label_ids_to_text(ids):
        labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 
            'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        return [labels[int(i)] for i in ids]
        
    @staticmethod
    def visualize_images(batch, n_rows=1, n_cols=8):
        x, y = batch
        labels = FashionMNIST.label_ids_to_text(y)
        show_images(x.squeeze(1), n_rows, n_cols, titles=labels)
