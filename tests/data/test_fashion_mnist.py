from mini_transformer.data.fashion_mnist import FashionMNIST
from torch.utils.data import DataLoader


def test_fashion_mnist():
    train_set = FashionMNIST(train=True)
    eval_set = FashionMNIST(train=False)
    assert len(train_set) == 60_000
    assert len(eval_set) == 10_000
    assert train_set[0][0].shape == (1, 28, 28)
    assert train_set[0][1] in range(10)


def test_fashion_mnist_batch():
    dataset = FashionMNIST()
    dataloader = DataLoader(dataset, batch_size=8)
    batch = next(iter(dataloader))
    FashionMNIST.visualize_images(batch, 1, 8)
