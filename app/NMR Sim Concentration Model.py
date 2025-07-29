import marimo

__generated_with = '0.14.13'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## NMR Sim Concentration Model
    - Taken using knowledge learn from NMR Sim Test.py, but cleaned up, and with certain aspects removed
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Define a spectrum creation and normalisation function

    """
    )
    return


app._unparsable_cell(
    r"""
    from nmrsim import SpinSystem
    from nmrsim.math import normalize_peaklist as normalize
    import numpy as np

    def spectrum_creator(analytes)
        \"\"\"
        Inputs:
            analytes: [
                {
                    v, array, frequencies in Hz
                    j, array, matrix of coupling constants
                    protons, int, number of contributing protons
                    concentration, float, concentration relative to each other
                }
            ]

        Outputs:
            x, y
        \"\"\"

        spectra = []
        for analyte in analytes:
            system = SpinSystem(analyte['v'], analyte['j']))
            normalized = normalize(SpinSystem.peaklist(), n=analyte['protons'])
            scaled = [(f, i * analyte['concentration']) for f, i in normalized]
            spectra.append(mplplot(scaled))

        mixed_peaks = np.concatenate(spectra, axis=1)
        
        return mixed_peaks

    """,
    name='_',
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, try and create and plot citrate via this""")
    return


app._unparsable_cell(
    r"""
    def createcitrate():
        v = np.array(
            [np.mean([2.95, 2.91]) * 600, np.mean([2.77, 2.73]) * 600]
        )
        J = np.zeros((2, 2))
        J[0, 1] = 15.89 / 400 * 600
        J[1, 0] = 15.89 / 400 * 600
        J = J + J.T
        return v, J

    def createdss():
        \"\"\"Create DSS reference peak at 0 ppm (0 Hz at any field strength)\"\"\"
        v = np.array([0.0])  # DSS peak at 0 Hz
        J = np.zeros((1, 1))  # No coupling for DSS
        return v, J


    spectrum = spectrum_creator(
        analytes = [
            {
                v, j = createdss(),
                protons = 9,
                concentration = 1.0
            },
            {
                v, j = createcitrate(),
                protons = 4,
                concentration = 0.1
            }
        ]
    )
    """,
    name='_',
)


if __name__ == '__main__':
    app.run()
