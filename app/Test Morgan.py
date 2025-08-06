import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import cProfile
    import pstats
    from morgan.createTrainingData import createTrainingData

    return cProfile, pstats


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
    return


@app.cell
def _(cProfile):
    sampleNumber = 10

    cProfile.run(
        'createTrainingData(substanceSpectrumIds=substanceSpectrumIds, sampleNumber=sampleNumber, scale=0.5)',
        'morgan/output.prof',
    )

    return


@app.cell
def _(pstats):
    with open('morgan/results.txt', 'w') as f:
        stats = pstats.Stats('morgan/output.prof', stream=f)
        stats.sort_stats('tottime')
        # stats.sort_stats('cumulative')
        # stats.sort_stats('ncalls')
        stats.print_stats()
    return


if __name__ == "__main__":
    app.run()
