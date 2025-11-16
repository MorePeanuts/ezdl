import torch
import argparse
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from mini_transformer.data.synthetic_data import SyntheticSineData
from mini_transformer.device_utils import get_single_device
from mini_transformer.trainer import train_regression_model_simple
from mini_transformer.plot_utils import plot_data_points


def train_linear_regression_on_sine_data():
    lr = 0.01
    num_epochs = 5
    device = get_single_device('cpu')
    model = torch.nn.Linear(in_features=4, out_features=1)
    data = SyntheticSineData()
    model.to(device)
    loss = MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_size = 600
    train_data, eval_data = train_test_split(data, train_size=train_size, shuffle=False)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)  # type: ignore
    eval_dataloader = DataLoader(eval_data, batch_size=16, shuffle=False, drop_last=False)  # type: ignore
    train_regression_model_simple(
        model, train_dataloader, eval_dataloader, optimizer, loss, device, num_epochs, eval_freq=8
    )
    return model, data, train_size


def show_onestep_predictions():
    model, data, _ = train_linear_regression_on_sine_data()
    onestep_preds = model(data[:][0]).detach().numpy()
    plot_data_points(
        data.time[data.window_size :],
        [data.labels, onestep_preds],
        xlabel='time',
        ylabel='x',
        legend=['labels', '1-step preds'],
        figsize=(12, 6),
    )


def show_multistep_predictions():
    model, data, train_size = train_linear_regression_on_sine_data()
    onestep_preds = model(data[:][0]).detach().numpy()
    multistep_preds = torch.zeros(data.total_steps)
    multistep_preds[:] = data.x

    for i in range(train_size + data.window_size, data.total_steps):
        multistep_preds[i] = model(multistep_preds[i - data.window_size : i].reshape((1, -1)))
    multistep_preds = multistep_preds.detach().numpy()

    plot_data_points(
        [data.time[data.window_size :], data.time[train_size + data.window_size :]],
        [onestep_preds, multistep_preds[train_size + data.window_size :]],
        xlabel='time',
        ylabel='x',
        legend=['1-step preds', 'multistep-preds'],
        figsize=(12, 6),
    )


def show_k_step_pred():
    model, data, _ = train_linear_regression_on_sine_data()

    def k_step_pred(k):
        features = []
        for i in range(data.window_size):
            features.append(data.x[i : i + data.total_steps - data.window_size - k + 1])
        for i in range(k):
            preds = model(torch.stack(features[i : i + data.window_size], 1))
            features.append(preds.reshape(-1))
        return features[data.window_size :]

    steps = (1, 4, 16, 64)
    preds = k_step_pred(steps[-1])
    plot_data_points(
        data.time[data.window_size + steps[-1] - 1 :],
        [preds[k - 1].detach().numpy() for k in steps],
        xlabel='time',
        ylabel='x',
        legend=[f'{k}-step preds' for k in steps],
        figsize=(12, 6),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A script to show auto-regression model prediction.'
    )
    tag = parser.add_mutually_exclusive_group(required=True)
    tag.add_argument('--onestep-preds', action='store_true')
    tag.add_argument('--multistep-preds', action='store_true')
    tag.add_argument('--k-step-preds', action='store_true')

    args = parser.parse_args()

    if args.onestep_preds:
        show_onestep_predictions()
    elif args.multistep_preds:
        show_multistep_predictions()
    elif args.k_step_preds:
        show_k_step_pred()
