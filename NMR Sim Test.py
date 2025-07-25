import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Define frequiences for vinyl group of vinyl acetate""")
    return


@app.cell
def _():
    import numpy as np

    def rioux():
        v = np.array([430.0, 265.0, 300.0])
        J = np.zeros((3, 3))
        J[0, 1] = 7.0
        J[0, 2] = 15.0
        J[1, 2] = 1.50
        J = J + J.T
        return [v, J]

    vinyl = rioux()
    print('v: ', vinyl[0])  # frequencies in Hz
    print('J: \n', vinyl[1])  # matrix of coupling constants
    return np, vinyl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Use the SpinSystem to create the peak list, and get the peak list""")
    return


@app.cell
def _(vinyl):
    from nmrsim import SpinSystem

    vinylsystem = SpinSystem(vinyl[0], vinyl[1])

    vinylsystem.peaklist()
    return SpinSystem, vinylsystem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot using nmrsim""")
    return


@app.cell
def _(vinylsystem):
    from nmrsim.plt import mplplot

    mplplot(vinylsystem.peaklist(), y_max=0.2)
    return (mplplot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, try for citrate

    J values taken from [This site](https://www.researchgate.net/figure/Citric-acid-1-H-NMR-spectra-Two-doublets-sloped-roofing-with-a-coupling-constant_fig14_354696479)
    """
    )
    return


@app.cell
def _(np):
    def createcitrate():
        v = np.array(
            [np.mean([2.95, 2.91]) * 600, np.mean([2.77, 2.73]) * 600]
        )
        J = np.zeros((2, 2))
        J[0, 1] = 15.89 / 400 * 600
        J[1, 0] = 15.89 / 400 * 600
        J = J + J.T
        return v, J

    citrate = createcitrate()
    print('v: ', citrate[0])  # frequencies in Hz
    print('J: \n', citrate[1])  # matrix of coupling constants
    return (citrate,)


@app.cell
def _(SpinSystem, citrate):
    citratesystem = SpinSystem(citrate[0], citrate[1], )

    citratesystem.peaklist()
    return (citratesystem,)


@app.cell
def _(citratesystem, mplplot):
    mplplot(citratesystem.peaklist(), y_max=0.4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mixing the two together""")
    return


@app.cell
def _(citratesystem, mplplot, vinylsystem):
    from nmrsim import Spectrum

    mix = Spectrum([citratesystem, vinylsystem])
    mplplot(mix.peaklist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Extract data of citrate graph into x, y""")
    return


@app.cell
def _(citratesystem, mplplot):
    import matplotlib.pyplot as plt

    x, y = mplplot(citratesystem.peaklist(), y_max=0.4)

    return plt, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create 100 noisy spectra""")
    return


@app.cell
def _(np, plt, x, y):
    ys = [y] * 100


    for i in range(0, len(ys)):
        sigma = np.random.rand() / 100
        noise = np.random.normal(0.0, sigma, len(ys[0]))
        ys[i] = y + noise
    
    plt.plot(x, ys[10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(mo):
    mo.md(r"""Create a spectra creator function""")
    return


@app.cell
def _(mplplot, np):
    def spectra_creator(plot):
        x, y = mplplot(plot.peaklist(), y_max=0.4)

        ys = [y] * 100
    
        for i in range(0, len(ys)):
            sigma = np.random.rand() / 100
            noise = np.random.normal(0.0, sigma, len(ys[0]))
            ys[i] = y + noise

        spectra = [[x, y] for y in ys]
        return spectra
    return (spectra_creator,)


@app.cell
def _(mo):
    mo.md(r"""Create 100 noisy spectra for both citrate and vinyl""")
    return


@app.cell
def _(citratesystem, plt, spectra_creator, vinylsystem):
    citrate_spectra = spectra_creator(citratesystem)
    vinyl_spectra = spectra_creator(vinylsystem)

    plt.plot(citrate_spectra[0][0], citrate_spectra[0][1])
    return citrate_spectra, vinyl_spectra


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    And now create a torch model to classify between the two?

    """
    )
    return


@app.cell
def _():
    import torch.nn as nn

    # Define the model
    model = nn.Sequential(
        nn.Linear(8, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )
    return (model,)


@app.cell(hide_code=True)
def _(citrate_spectra, np, vinyl_spectra):
    import torch
    from sklearn.model_selection import train_test_split



    data = np.concatenate([citrate_spectra, vinyl_spectra], axis=0)

    labels = np.concatenate([
        np.zeros(100),
        np.ones(100)
    ])

    train_ratio = 0.7   # 70% of the data is used for training, 30% for testing

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, train_size=train_ratio, shuffle=True
    )

    data_train = torch.tensor(citrate_spectra)
    labels_train = torch.tensor(vinyl_spectra)

    print(data_train.shape)
    print(labels_train.shape)
    return data_test, data_train, labels_test, labels_train, torch


@app.cell
def _(
    data_test,
    data_train,
    labels_test,
    labels_train,
    loss_fn,
    model,
    np,
    optimizer,
    torch,
):
    n_epochs = 100   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(data_train), batch_size)

    import tqdm
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
                data_batch = data_train[start : start + batch_size]
                labels_batch = labels_train[start : start + batch_size]
                # forward pass
                labels_pred = model(data_batch)
                loss = loss_fn(labels_pred, labels_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        labels_pred = model(data_test)
        mse = loss_fn(labels_pred, labels_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
    return


if __name__ == "__main__":
    app.run()
