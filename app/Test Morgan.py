import marimo

__generated_with = '0.14.17'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        rf"""Code block to profile Morgan's code from where I was converting numpy to jax."""
    )
    return


@app.cell
def _():
    import cProfile
    import pstats
    from morgancode.createTrainingData import createTrainingData
    import jax

    # from jax.config import config

    # config.update('jax_platform_name', 'rocm')
    print(jax.devices())
    return cProfile, createTrainingData, pstats


@app.cell
def _():
    substanceDict = {
        'L-Alanine': ['SP:3208'],
        'L-Arginine': ['SP:3212', 'SP:3285', 'SP:3360', 'SP:3388'],
        'L-Asparagine': ['SP:3408', 'SP:3597'],
        'L-Aspartic_Acid': ['SP:3526', 'SP:3603'],
        'L-Cysteine': ['SP:3723', 'SP:3725'],
        'L-Glutamic_Acid': ['SP:3412', 'SP:3690'],
        'L-Glutamine': ['SP:3108'],
        'L-Histidine': ['SP:3099', 'SP:3684'],
        'L-Isoleucine': ['SP:3390', 'SP:3502'],
        'L-Leucine': ['SP:3551'],
        'L-Lysine': ['SP:3506', 'SP:3560'],
        'L-Methionine': ['SP:3456', 'SP:3509'],
        'L-Proline': ['SP:3140', 'SP:3406'],
        'L-Phenylalanine': ['SP:3326', 'SP:3462', 'SP:3507'],
        'L-Serine': ['SP:3324', 'SP:3427', 'SP:3732'],
        'L-Threonine': ['SP:3327', 'SP:3437'],
        'L-Tryptophan': ['SP:3342', 'SP:3455'],
        'L-Tyrosine': ['SP:3464'],
        'L-Valine': ['SP:3413', 'SP:3490'],
        'Glycine': ['SP:3365', 'SP:3682'],
    }
    substanceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]
    return (substanceSpectrumIds,)


@app.cell
def _(cProfile, createTrainingData, substanceSpectrumIds):
    sampleNumber = 100

    _ = createTrainingData(
        substanceSpectrumIds=substanceSpectrumIds, sampleNumber=1
    )   # Prebuild before everything

    cProfile.run(
        'createTrainingData(substanceSpectrumIds=substanceSpectrumIds, sampleNumber=sampleNumber, scale=0.5)',
        'morgan/output.prof',
    )

    dumbvar = True   # So that the next cell always runs when this is rerun

    return (dumbvar,)


@app.cell
def _(dumbvar, pstats):
    if dumbvar:
        pass

    with open('morgan/results.txt', 'w') as f:
        stats = pstats.Stats('morgan/output.prof', stream=f)
        stats.sort_stats('tottime')
        # stats.sort_stats('cumulative')
        # stats.sort_stats('ncalls')
        stats.print_stats()
    return


@app.cell
def _(createTrainingData, substanceSpectrumIds):
    # Ensure the plot is reasonable
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_config import setup_dark_theme, save_figure, get_colors

    substance = createTrainingData(
        substanceSpectrumIds=substanceSpectrumIds, sampleNumber=1, scale=0.5
    )

    import matplotlib.pyplot as plt

    # Apply dark theme
    setup_dark_theme()
    colors = get_colors(1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(substance['positions'], substance['intensities'][0], color=colors[0], linewidth=2)
    plt.title('Training Data Sample', fontsize=14)
    plt.xlabel('Chemical Shift (ppm)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.grid(True, alpha=0.3)
    save_figure(plt.gcf(), 'test_morgan_sample.png')
    return


if __name__ == '__main__':
    app.run()
