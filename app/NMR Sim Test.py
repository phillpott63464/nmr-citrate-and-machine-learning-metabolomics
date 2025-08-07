import marimo

__generated_with = '0.14.16'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    from sklearn.model_selection import train_test_split
    import torch

    return mo, torch, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Define frequiences for vinyl group of vinyl acetate""")
    return


@app.cell
def _():
    import numpy as np

    def rioux():   # original vinyl
        v = np.array([430.0, 265.0, 300.0])
        J = np.zeros((3, 3))
        J[0, 1] = 7.0
        J[0, 2] = 15.0
        J[1, 2] = 1.50
        J = J + J.T
        return [v, J]

    def createfalsecitrate():
        v = np.array([2.85 * 600, 2.76 * 600])
        J = np.zeros((2, 2))
        J[0, 1] = 15.89 / 400 * 600
        J[1, 0] = 15.89 / 400 * 600
        J = J + J.T
        return v, J

    vinyl = createfalsecitrate()
    print('v: ', vinyl[0])  # frequencies in Hz
    print('J: \n', vinyl[1])  # matrix of coupling constants
    return np, vinyl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Use the SpinSystem to create the peak list, and get the peak list"""
    )
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
    citratesystem = SpinSystem(
        citrate[0],
        citrate[1],
    )

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
    mo.md(r"""Create 10000 noisy spectra""")
    return


@app.cell
def _(np, plt, x, y):
    ys = [y] * 10000

    for i in range(0, len(ys)):
        sigma = np.random.rand() / 50
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
        ys = [y] * 10000

        for i in range(0, len(ys)):
            # Much more realistic noise levels
            base_noise_level = (
                0.005 + np.random.rand() * 0.01
            )  # 0.5-1.5% noise

            # 1. Gentle Gaussian noise
            gaussian_noise = np.random.normal(
                0.0, base_noise_level, len(ys[0])
            )

            # 2. Subtle baseline drift
            baseline_drift = np.random.normal(0, 0.002) * np.linspace(
                -1, 1, len(ys[0])
            )

            # 3. Occasional small spikes (much less frequent)
            spike_noise = np.zeros_like(ys[0])
            if np.random.rand() > 0.8:  # Only 20% chance of spikes
                n_spikes = np.random.poisson(1)  # Usually 0-2 spikes
                spike_positions = np.random.choice(
                    len(ys[0]), size=min(n_spikes, len(ys[0])), replace=False
                )
                spike_noise[spike_positions] = np.random.normal(
                    0, base_noise_level * 3, len(spike_positions)
                )

            # 4. Small intensity variation
            scale_factor = 1.0 + np.random.normal(
                0, 0.05
            )  # ±5% intensity variation

            # 5. Minimal phase distortion
            phase_error = np.random.normal(0, 0.02)
            y_phase_distorted = y * np.cos(phase_error) + np.random.normal(
                0, 0.001, len(y)
            ) * np.sin(phase_error)

            total_noise = gaussian_noise + baseline_drift + spike_noise
            ys[i] = (y_phase_distorted * scale_factor) + total_noise

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
    # plt.plot(vinyl_spectra[0][0], vinyl_spectra[0][1])
    return citrate_spectra, vinyl_spectra


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And now create a torch model to classify between the two?""")
    return


@app.cell
def _():
    import torch.nn as nn

    # Define the model
    model = nn.Sequential(
        nn.Linear(800, 24),  # 800*24
        nn.ReLU(),
        nn.Linear(24, 12),  # 24*12
        nn.ReLU(),
        nn.Linear(12, 6),  # 12*6
        nn.ReLU(),
        nn.Linear(6, 1),  # 6*1
    )
    return model, nn


@app.cell
def _(
    citrate_spectra,
    np,
    torch,
    train_ratio,
    train_test_split,
    vinyl_spectra,
):
    citrate_y = np.array([spectrum[1] for spectrum in citrate_spectra])
    vinyl_y = np.array([spectrum[1] for spectrum in vinyl_spectra])

    data = np.concatenate([citrate_y, vinyl_y], axis=0)

    labels = np.concatenate(
        [np.zeros(int(len(data) / 2)), np.ones(int(len(data) / 2))]
    )

    print(data.shape)
    print(labels.shape)

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, train_size=train_ratio, shuffle=True
    )

    data_train = torch.tensor(data_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)

    print(data_train.shape)
    print(labels_train.shape)

    return data_test, data_train, labels_test, labels_train


@app.cell
def _(mo):
    mo.md(
        rf"""
    Attempted PCA on NMR spectra

    Result: absoloute nonsense, 99% of the variance is not contained in a single data point
    """
    )
    return


@app.cell
def _(citrate_spectra, data_train, plt):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=400)

    data_train_reduced = pca.fit_transform(data_train)

    print(data_train.shape)
    print(data_train_reduced.shape)

    print(data_train_reduced[400])

    plt.plot(data_train[400])
    plt.plot(data_train_reduced[400])
    plt.plot(citrate_spectra[400][0], data_train[400])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Attempted binning

    It works much better for containing the shape of the data, but since the integral is summed, it completely screws with intensities and ergo concentration

    Using max instead of sum keeps peak heights, still messes with total integral area
    """
    )
    return


@app.cell
def _(citrate_spectra, np, plt, vinyl_spectra):
    # Method 1: Spectral Binning (most common for NMR)
    def spectral_binning(spectra, n_bins=50):
        """Bin spectra into regions and integrate each bin"""
        x = spectra[0][0]  # frequency axis
        bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)

        binned_data = []
        for spectrum in spectra:
            y = spectrum[1]
            binned_spectrum = []

            for i in range(len(bin_edges) - 1):
                # Find points in this bin
                mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
                # Integrate (sum) the intensity in this bin
                bin_integral = np.max(y[mask]) if np.any(mask) else 0
                binned_spectrum.append(bin_integral)

            binned_data.append(binned_spectrum)

        return np.array(binned_data)

    # Apply binning
    citrate_binned = spectral_binning(citrate_spectra, n_bins=50)
    vinyl_binned = spectral_binning(vinyl_spectra, n_bins=50)

    print(f'Original spectrum length: {len(citrate_spectra[0][1])}')
    print(f'Binned spectrum length: {citrate_binned.shape[1]}')
    print(
        f'Dimension reduction: {len(citrate_spectra[0][1]) / citrate_binned.shape[1]:.1f}x'
    )

    print(f'Max intensity of original spectra: {max(citrate_spectra[0][1])}')
    print(f'Max intensity of binned spectra: {max(citrate_binned[0])}')

    # Plot comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(citrate_spectra[0][0], citrate_spectra[0][1], label='Original')
    plt.title('Original Spectrum')
    plt.xlabel('Frequency (Hz)')

    plt.subplot(1, 2, 2)
    plt.plot(
        range(len(citrate_binned[0])), citrate_binned[0], 'o-', label='Binned'
    )
    plt.title('Binned Spectrum')
    plt.xlabel('Bin Number')

    plt.tight_layout()
    plt.show()

    return citrate_binned, vinyl_binned


@app.cell
def _(citrate_binned, np, vinyl_binned):
    # Use binned data for classification instead of raw spectra
    binned_data = np.concatenate([citrate_binned, vinyl_binned], axis=0)
    binned_labels = np.concatenate(
        [np.zeros(len(citrate_binned)), np.ones(len(vinyl_binned))]
    )

    print(f'Binned data shape: {binned_data.shape}')
    print(f'Labels shape: {binned_labels.shape}')

    # This should work much better than PCA!
    return binned_data, binned_labels


@app.cell
def _(binned_data, binned_labels, nn, torch, train_test_split):
    # Use binned data instead of raw spectra
    train_ratio = 0.7

    (
        data_train_binned,
        data_test_binned,
        labels_train_binned,
        labels_test_binned,
    ) = train_test_split(
        binned_data, binned_labels, train_size=train_ratio, shuffle=True
    )

    data_train_binned = torch.tensor(data_train_binned, dtype=torch.float32)
    labels_train_binned = torch.tensor(
        labels_train_binned, dtype=torch.float32
    )
    data_test_binned = torch.tensor(data_test_binned, dtype=torch.float32)
    labels_test_binned = torch.tensor(labels_test_binned, dtype=torch.float32)

    print(f'Binned training data shape: {data_train_binned.shape}')
    print(f'Binned training labels shape: {labels_train_binned.shape}')

    # Create a new model with the correct input size (50 bins instead of 800)
    model_binned = nn.Sequential(
        nn.Linear(50, 24),  # 50 bins input
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )

    return (
        data_test_binned,
        data_train_binned,
        labels_test_binned,
        labels_train_binned,
        model_binned,
        train_ratio,
    )


app._unparsable_cell(
    r"""
    {
            'scales': pd.DataFrame(scalesDict),
            'positions': positions,
            'intensities': np.vstack(intensitiesList), 
            'components': np.vstack(untransformedComponentsList)
            }import tqdm
    import copy
    import torch.optim as optim

    def train_model(
        model,
        data_train,
        labels_train,
        data_test,
        labels_test,
        n_epochs=100,
        batch_size=10,
        lr=0.001,
    ):
        \"\"\"Generic training function for binary classification\"\"\"

        batch_start = torch.arange(0, len(data_train), batch_size)

        # loss function and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = np.inf
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
                    loss = loss_fn(labels_pred.squeeze(), labels_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(loss=float(loss))

            # evaluate accuracy at end of each epoch
            model.eval()
            with torch.no_grad():
                labels_pred = model(data_test)
                test_loss = loss_fn(labels_pred.squeeze(), labels_test)
                test_loss = float(test_loss)
                history.append(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_weights = copy.deepcopy(model.state_dict())

        # Final evaluation
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            labels_pred = model(data_test)
            predictions = torch.sigmoid(labels_pred) > 0.5
            accuracy = (predictions.squeeze() == labels_test).float().mean()

            print(f'Final Accuracy: {accuracy:.2%}')
            print(f'Best Loss: {best_loss:.4f}')

            # Show some example predictions
            probs = torch.sigmoid(labels_pred[:10]).squeeze()
            print(f'First 10 prediction probabilities: {probs}')
            print(f'First 10 true labels: {labels_test[:10]}')

        return best_loss, best_weights, history, accuracy

    """,
    name='_',
)


@app.cell
def _(
    data_test_binned,
    data_train_binned,
    labels_test_binned,
    labels_train_binned,
    model_binned,
    train_model,
):
    # Train the binned model
    print('Training binned model...')
    (
        best_loss_binned,
        best_weights_binned,
        history_binned,
        accuracy_binned,
    ) = train_model(
        model_binned,
        data_train_binned,
        labels_train_binned,
        data_test_binned,
        labels_test_binned,
        n_epochs=100,
        batch_size=10,
        lr=0.001,
    )

    return accuracy_binned, best_loss_binned, history_binned


@app.cell
def _(data_test, data_train, labels_test, labels_train, model, train_model):
    # Train the raw data model for comparison
    print('Training raw data model...')
    best_loss_raw, best_weights_raw, history_raw, accuracy_raw = train_model(
        model,
        data_train,
        labels_train,
        data_test,
        labels_test,
        n_epochs=100,
        batch_size=10,
        lr=0.0001,  # Lower learning rate for the larger model
    )

    return accuracy_raw, best_loss_raw, history_raw


@app.cell
def _(history_binned, history_raw, plt):
    # Compare training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history_binned, label='Binned Data')
    plt.plot(history_raw, label='Raw Data')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history_binned[10:], label='Binned Data')
    plt.title('Binned Model Loss (after epoch 10)')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history_raw[10:], label='Raw Data')
    plt.title('Raw Model Loss (after epoch 10)')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(accuracy_binned, accuracy_raw, best_loss_binned, best_loss_raw, mo):
    mo.md(
        f"""
    ## Model Comparison Results

    | Model | Accuracy | Best Loss |
    |-------|----------|-----------|
    | Binned Data (50 features) | {accuracy_binned:.2%} | {best_loss_binned:.4f} |
    | Raw Data (800 features) | {accuracy_raw:.2%} | {best_loss_raw:.4f} |

    The binned approach should perform significantly better due to:
    - Better feature representation
    - Reduced noise
    - More stable training
    - Lower computational cost
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create a DSS peak and integrate with compound spectra""")
    return


@app.cell
def _(SpinSystem, mplplot, np, plt):
    from nmrsim.math import normalize_peaklist as normalize

    def create_dss():
        """Create DSS reference peak at 0 ppm (0 Hz at any field strength)"""
        v = np.array([0.0])  # DSS peak at 0 Hz
        J = np.zeros((1, 1))  # No coupling for DSS
        return v, J

    dssvars = create_dss()

    dss_system = SpinSystem(dssvars[0], dssvars[1])

    dss_concentration = 1

    dss_normalized_peaklist = normalize(dss_system.peaklist(), n=9)
    dss_scaled = [
        (f, i * dss_concentration) for f, i in dss_normalized_peaklist
    ]

    dss_normal_x_y = mplplot(dss_system.peaklist())
    dss_normalized_x_y = mplplot(dss_normalized_peaklist)
    dss_scaled_x_y = mplplot(dss_scaled)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(dss_normal_x_y[0], dss_normal_x_y[1])
    plt.subplot(1, 3, 2)
    plt.plot(dss_normalized_x_y[0], dss_normalized_x_y[1])
    plt.subplot(1, 3, 3)
    plt.plot(dss_scaled_x_y[0], dss_scaled_x_y[1])

    return dss_scaled_x_y, normalize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Attempt to use first order multiplet""")
    return


@app.cell
def _(citrate_spectra, citratesystem, mplplot, normalize, plt):
    citrate_normalized_peaklist = normalize(citratesystem.peaklist(), n=4)

    citrate_scaled = [(f, i * 0.1) for f, i in citrate_normalized_peaklist]

    citrate_normalized_x_y = mplplot(citrate_normalized_peaklist, 0.5)
    citrate_scaled_x_y = mplplot(citrate_scaled, 0.5)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(citrate_spectra[0][1])
    plt.subplot(1, 3, 2)
    plt.plot(citrate_normalized_x_y[1])
    plt.subplot(1, 3, 3)
    plt.plot(citrate_scaled_x_y[1])
    return (citrate_scaled_x_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Combine scaled DSS peak with scaled citrate peaks""")
    return


@app.cell
def _(citrate_scaled_x_y, dss_scaled_x_y, np, plt):
    # mixed_peaks = np.concatenate((np.array(dss_scaled_x_y), np.array(citrate_scaled_x_y)), axis=1)
    mixed_peaks = np.concatenate((dss_scaled_x_y, citrate_scaled_x_y), axis=1)

    print(np.array(dss_scaled_x_y).shape, np.array(citrate_scaled_x_y).shape)
    print(mixed_peaks.shape)

    plt.plot(mixed_peaks[0], mixed_peaks[1])

    return (mixed_peaks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Test resampling, which is better than binning for retaining peak shape and integrals

    Nope, no it's not, completely destroys both intensities and definition
    """
    )
    return


@app.cell
def _(mixed_peaks, np, plt):
    def resample_spectrum(spectrum, n_points=256):
        x_old, y_old = spectrum
        x_new = np.linspace(x_old.min(), x_old.max(), n_points)
        y_new = np.interp(x_new, x_old, y_old)
        return x_new, y_new

    resampled_mixed_peaks = resample_spectrum(mixed_peaks)

    print(np.array(resampled_mixed_peaks).shape)

    plt.plot(resampled_mixed_peaks[0], resampled_mixed_peaks[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Try just yoinking parts of the plot out""")
    return


@app.cell
def _(mixed_peaks, np, plt):
    yoinkrange = [
        [1600, 1800],
        [-5, 5],
        [500, 700],
    ]

    indices_range = [
        np.where(
            (mixed_peaks[0] >= yoinkrange[0][0])
            & (mixed_peaks[0] <= yoinkrange[0][1])
        )[0],
        np.where(
            (mixed_peaks[0] >= yoinkrange[1][0])
            & (mixed_peaks[0] <= yoinkrange[1][1])
        )[0],
    ]

    print(indices_range)

    print(mixed_peaks.shape)

    yoinked_peaks = np.hstack(
        [
            (mixed_peaks[:, indices_range[0]]),
            (mixed_peaks[:, indices_range[1]]),
        ]
    )

    print(yoinked_peaks.shape)

    plt.plot(yoinked_peaks[0], yoinked_peaks[1])
    return


if __name__ == '__main__':
    app.run()
