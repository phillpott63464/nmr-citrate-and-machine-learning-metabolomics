import marimo

__generated_with = '0.14.16'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Code taken from [This website](https://machinelearningmastery.com/building-a-regression-model-in-pytorch/) and adapted into marimo notebook format for testing."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Import the dataset and print feature names""")
    return


@app.cell
def _():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    print(data.feature_names)
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""- Extra data and targets from dataset""")
    return


@app.cell
def _(data):
    X, y = data.data, data.target
    print(f'X shape: {X.shape}')
    print(f'X data: {X[0]}')
    print(f'y shape: {y.shape}')
    print(f'y data: {y[0]}')
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""- Define a model. This is a 4 layer MLP using ReLU activation."""
    )
    return


@app.cell
def _():
    import torch.nn as nn
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    model = nn.Sequential(
        nn.Linear(8, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    ).to(device)
    return device, model, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""- Define a loss function (mean square error) and optimizer (adam)"""
    )
    return


@app.cell
def _(model, nn):
    import torch.optim as optim

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Split the training data into test and train datasets
    - Convert into tensors
    - Reshape y data
    """
    )
    return


@app.cell
def _(X, device, torch, y):
    from sklearn.model_selection import train_test_split

    train_ratio = 0.7   # 70% of the data is used for training, 30% for testing

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, shuffle=True
    )
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = (
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    )
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = (
        torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Define training parameters""")
    return


@app.cell
def _(X_train, torch):
    n_epochs = 100   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    return batch_size, batch_start, n_epochs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Run training loop and define parameters to hold values""")
    return


@app.cell
def _(
    X_test,
    X_train,
    batch_size,
    batch_start,
    loss_fn,
    model,
    n_epochs,
    optimizer,
    y_test,
    y_train,
):
    import tqdm
    import numpy as np
    import copy

    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(
            batch_start, unit='batch', mininterval=0, disable=True
        ) as bar:
            bar.set_description(f'Epoch {epoch}')
            for start in bar:
                # take a batch
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
    return best_mse, best_weights, history, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Reload the best model from trained weights""")
    return


@app.cell
def _(best_weights, model):
    model.load_state_dict(best_weights)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Print the best trial MSE and RMSE (Root of MSE)
    - Plot A graph of MSE/Epochs
    """
    )
    return


@app.cell
def _(best_mse, history, np):
    import matplotlib.pyplot as plt

    print('MSE: %.2f' % best_mse)
    print('RMSE: %.2f' % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == '__main__':
    app.run()
