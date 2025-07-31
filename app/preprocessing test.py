import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    # Store device info for markdown display
    hip_version = torch.version.hip
    cuda_built = torch.backends.cuda.is_built()
    gpu_count = torch.cuda.device_count()

    return cuda_built, gpu_count, hip_version, mo


@app.cell(hide_code=True)
def _(cuda_built, gpu_count, hip_version, mo):
    mo.md(
        f"""
    ## PyTorch GPU Setup Information

    - **HIP Version:** {hip_version}
    - **CUDA Built:** {cuda_built}
    - **GPU Device Count:** {gpu_count}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# NMR Sim Concentration Model""")
    return


@app.cell(hide_code=True)
def _(mo, spectrafigures, substanceDict):
    mo.md(
        rf"""
    ## Data Creation

    Create spectra using morgan's code overlaying nmrsim

    Contains metabolites: {''.join(f"{x}, " for x in substanceDict)}

    {mo.as_html(spectrafigures)}
    """
    )
    return


@app.cell
def _():
    # Data generation

    from morgan.createTrainingData import createTrainingData
    import numpy as np

    substanceDict = {
        'Citric acid': ['SP:3368'],
        'Succinic acid': ['SP:3211'],
        'Maleic acid': ['SP:3110'],
        'Lactic acid': ['SP:3675'],
    }

    substanceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]
    count = 1000

    # Use the built-in loop capability of createTrainingData
    batch_data = createTrainingData(
        substanceSpectrumIds=substanceSpectrumIds,
        sampleNumber=count,  # This creates 1000 samples in one call
        rondomlyScaleSubstances=True,
        # referenceSubstanceSpectrumId='dss',
    )

    # Extract data into your current format
    labels = []
    spectra = []

    for i in range(count):
        # Extract scales for this sample
        sample_scales = {
            key: [values[i]] for key, values in batch_data['scales'].items()
        }

        # Create spectrum dict in your current format
        spectrum = {
            'scales': sample_scales,
            'intensities': batch_data['intensities'][
                i : i + 1
            ],  # Keep 2D shape
            'positions': batch_data['positions'],
            'components': batch_data[
                'components'
            ],  # This is shared across all samples
        }
        spectra.append(spectrum)

    print(
        ''.join(f"{x['scales']}\n'" for x in spectra[:5])
    )   # Show first 5 only
    print(spectra[0]['intensities'].shape)   # Y axis
    print(spectra[0]['positions'].shape)   # X axis
    print(spectra[0]['components'].shape)   # Peaks of all separate components

    # Get sample information for markdown display
    sample_scales_preview = '\n'.join(
        [
            f"Sample {i+1}: {spectrum['scales']}"
            for i, spectrum in enumerate(spectra[:5])
        ]
    )
    intensities_shape = spectra[0]['intensities'].shape
    positions_shape = spectra[0]['positions'].shape
    components_shape = spectra[0]['components'].shape

    return (
        components_shape,
        count,
        createTrainingData,
        intensities_shape,
        labels,
        np,
        positions_shape,
        sample_scales_preview,
        spectra,
        substanceDict,
        substanceSpectrumIds,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(
    components_shape,
    count,
    intensities_shape,
    mo,
    positions_shape,
    sample_scales_preview,
):
    mo.md(
        rf"""
    ## Data Generation Results

    **Generated {count} samples successfully**

    **Data Structure:**

    - **Intensities Shape:** {intensities_shape} (Y axis)

    - **Positions Shape:** {positions_shape} (X axis) 

    - **Components Shape:** {components_shape} (Peaks of all separate components)

    **Sample Scales Preview (First 5 samples):**
    ```
    {sample_scales_preview}
    ```
    """
    )
    return


@app.cell
def _(labels, spectra):
    import matplotlib.pyplot as plt

    print(len(spectra))
    print(len(labels))

    graph_count = 3

    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter)
        # plt.title(f'{round(labels[graphcounter], 3)}M')
        plt.plot(
            spectra[graphcounter]['positions'],
            spectra[graphcounter]['intensities'][0],
        )

    spectrafigures = plt.gca()
    return graph_count, plt, spectrafigures


@app.cell(hide_code=True)
def _(mo, spectra):
    mo.md(
        rf"""
    ## Spectra Overview

    - **Total Spectra Generated:** {len(spectra)}
    """
    )
    return


@app.cell
def _(mo, referencefigure):
    mo.md(
        rf"""
    ## Extract Reference Spectra
    Extract spectra of individual components as a reference for the model

    {mo.as_html(referencefigure)}
    """
    )
    return


@app.cell
def _(createTrainingData, plt, spectra, substanceDict, substanceSpectrumIds):
    reference_spectra_raw = createTrainingData(
        substanceSpectrumIds=substanceSpectrumIds,
        sampleNumber=1,  # This creates 1000 samples in one call
        rondomlyScaleSubstances=False,
        # referenceSubstanceSpectrumId='dss',
    )

    reference_spectra = {
        substanceDict[substance][0]: reference_spectra_raw['components'][index]
        for index, substance in enumerate(substanceDict)
    }

    print(reference_spectra)

    for substance in substanceDict:
        plt.plot(
            spectra[0]['positions'],
            reference_spectra[substanceDict[substance][0]],
        )

    referencefigure = plt.gca()

    return reference_spectra, referencefigure


@app.cell
def _(mo, preprocessedfigure, preprocessedreferencefigure):
    mo.md(
        rf"""
    ## Spectra Preprocessing
    - Extract only the relevant parts of the spectra
    - Add baseline distortion
    - Extract ratio of citrate to reference

    {mo.as_html(preprocessedfigure)}

    {mo.as_html(preprocessedreferencefigure)}
    """
    )
    return


@app.cell
def _(np, plt, reference_spectra, spectra, substanceDict):
    # Data preprocessing

    from scipy.signal import resample
    from scipy.interpolate import interp1d
    from multiprocessing import Pool
    import os
    import functools

    def preprocess_peaks(
        intensities,
        positions,
        ranges=[[-100, 100]],
        baseline_distortion=False,
        downsample=0,
    ):
        indices_range = []
        for range_item in ranges:  # Changed from 'range' to 'range_item'
            indices_range.append(
                np.where(
                    (positions >= range_item[0]) & (positions <= range_item[1])
                )[0]
            )

        new_intensities = []
        new_positions = []

        for (
            range_indices
        ) in indices_range:  # Changed from 'range' to 'range_indices'
            # Fix: properly extract intensities - flatten the 2D array for this range
            temp_intensities = intensities[
                range_indices
            ]  # Take first row and specified indices
            temp_positions = positions[
                range_indices
            ]  # Direct indexing for positions

            # Add baseline distortion if enabled
            if baseline_distortion:
                # Normalize positions to [0, 1] for consistent baseline calculation
                if len(temp_positions) > 1:  # Avoid division by zero
                    x_normalized = (
                        temp_positions - np.min(temp_positions)
                    ) / (np.max(temp_positions) - np.min(temp_positions))
                    baseline = 0.02 * np.sin(0.5 * np.pi * x_normalized)
                    temp_intensities = temp_intensities + baseline

            new_intensities = np.concatenate(
                [new_intensities, temp_intensities]
            )
            new_positions = np.concatenate([new_positions, temp_positions])

        if downsample > 0:
            # For NMR data, positions are typically in descending order
            # Sort the data by position to ensure proper binning
            sorted_indices = np.argsort(new_positions)
            new_positions = new_positions[sorted_indices]
            new_intensities = new_intensities[sorted_indices]

            min_pos, max_pos = new_positions.min(), new_positions.max()
            bin_edges = np.linspace(min_pos, max_pos, downsample + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            downsampled_intensities = np.zeros(downsample)

            for i in range(downsample):
                mask = (new_positions >= bin_edges[i]) & (
                    new_positions < bin_edges[i + 1]
                )
                if np.any(mask):
                    # Use interpolation to better preserve peak shapes
                    # This approach maintains area while giving better peak representation
                    bin_width = bin_edges[i + 1] - bin_edges[i]
                    if len(new_positions[mask]) > 1:
                        # Use trapezoidal integration but normalize by actual data spacing
                        area = np.trapz(
                            new_intensities[mask], new_positions[mask]
                        )
                        downsampled_intensities[i] = area / bin_width
                    else:
                        # Single point - just use the intensity value
                        downsampled_intensities[i] = new_intensities[mask][0]
                else:
                    downsampled_intensities[i] = 0.0

            new_intensities = downsampled_intensities
            new_positions = bin_centers

        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        ratios = {
            substanceDict[substance][0]: scales[substanceDict[substance][0]][0]
            / scales['tsp'][0]
            for substance in substanceDict
        }

        return ratios

    def preprocess_spectra(
        spectra, ranges, substanceDict, baseline_distortion=False, downsample=0
    ):
        new_positions, new_intensities = preprocess_peaks(
            intensities=spectra['intensities'][0],
            positions=spectra['positions'],
            ranges=ranges,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
        )

        ratios = preprocess_ratio(spectra['scales'], substanceDict)

        return {
            'intensities': new_intensities,
            'positions': new_positions,
            'scales': spectra['scales'],
            'components': spectra['components'],
            'ratios': ratios,
        }

    ranges = [[-100, 100]]
    baseline_distortion = True
    downsample = int(2048)

    def process_spectra_parallel(spectra, n_processes=None):
        """Parallelize the preprocessing of spectra"""
        if n_processes is None:
            n_processes = os.cpu_count()

        # Create a partial function with fixed parameters
        process_func = functools.partial(
            preprocess_spectra,
            ranges=ranges,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample
        )

        with Pool(processes=n_processes) as pool:
            preprocessed_spectra = pool.map(process_func, spectra)

        return preprocessed_spectra

    preprocessed_spectra = process_spectra_parallel(spectra)

    def process_references(reference_spectra):
        temp = [
            preprocess_peaks(
                positions=spectra[0]['positions'],
                intensities=reference_spectra[spectrum],
                downsample=2048,
            )
            for spectrum in reference_spectra
        ]

        preprocessed_reference_spectra = {
            spectrum: temp[i][1]
            for i, spectrum in enumerate(reference_spectra)
        }

        return preprocessed_reference_spectra

    preprocessed_reference_spectra = process_references(reference_spectra)

    def generate_figure():

        plt.figure(figsize=(8, 4))
        for substance in substanceDict:
            plt.subplot(1, 2, 1)
            plt.plot(
                spectra[0]['positions'],
                reference_spectra[substanceDict[substance][0]],
            )
            plt.subplot(1, 2, 2)
            plt.plot(
                preprocessed_spectra[0]['positions'],
                preprocessed_reference_spectra[substanceDict[substance][0]],
            )

        return plt.gca()

    preprocessedreferencefigure = generate_figure()

    print(len(preprocessed_spectra[0]['positions']))
    print(len(preprocessed_spectra[0]['intensities']))

    positions_count = len(preprocessed_spectra[0]['positions'])
    intensities_count = len(preprocessed_spectra[0]['intensities'])
    return (
        baseline_distortion,
        intensities_count,
        positions_count,
        preprocessed_spectra,
        preprocessedreferencefigure,
        ranges,
    )


@app.cell(hide_code=True)
def _(baseline_distortion, intensities_count, mo, positions_count, ranges):
    mo.md(
        rf"""
    ## Preprocessing Results

    **Configuration:**

    - **Baseline Distortion:** {'Enabled' if baseline_distortion else 'Disabled'}

    - **Spectral Ranges:** {ranges}

    **Processed Data Dimensions:**

    - **Positions Count:** {positions_count}

    - **Intensities Count:** {intensities_count}
    """
    )
    return


@app.cell
def _(graph_count, labels, plt, preprocessed_spectra, spectra):
    print(len(spectra))
    print(len(labels))

    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter2 in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter2)
        # plt.title(f'{round(labels[graphcounter], 3)}M')
        plt.plot(
            spectra[graphcounter2]['positions'],
            spectra[graphcounter2]['intensities'][0],
        )
        plt.plot(
            preprocessed_spectra[graphcounter2]['positions'],
            preprocessed_spectra[graphcounter2]['intensities'],
        )

    preprocessedfigure = plt.gca()
    return (preprocessedfigure,)


if __name__ == "__main__":
    app.run()
