import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


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
    mo.md(r"""Define a spectrum creation and normalisation function""")
    return


@app.cell
def _():
    from nmrsim import SpinSystem
    from nmrsim.math import normalize_peaklist as normalize
    from nmrsim.plt import mplplot

    def spectrum_creator(analytes):
        """
        Inputs:
            analytes: [
                {
                    'v': array, frequencies in Hz
                    'j': array, matrix of coupling constants
                    'protons': int, number of contributing protons
                    'concentration': float, concentration relative to each other
                }
            ]

        Outputs:
            x, y arrays for plotting
        """

        all_peaklists = []
        for analyte in analytes:
            system = SpinSystem(analyte['v'], analyte['j'])
            normalized = normalize(system.peaklist(), n=analyte['protons'])
            scaled = [(f, i * analyte['concentration']) for f, i in normalized]
            all_peaklists.extend(scaled)

        # Now convert the combined peaklist to plotting format
        plots = mplplot(all_peaklists, hidden=True)
        return plots

    return (spectrum_creator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, try and create and plot citrate via this""")
    return


@app.cell
def _():
    import numpy as np

    def createcitrate():
        v = np.array(
            [np.mean([2.95, 2.91]) * 600, np.mean([2.77, 2.73]) * 600]
        )
        J = np.zeros((2, 2))
        J[0, 1] = 15.89 / 400 * 600
        J[1, 0] = 15.89 / 400 * 600
        return v, J

    def createdss():
        """Create DSS reference peak at 0 ppm (0 Hz at any field strength)"""
        v = np.array([0.0])  # DSS peak at 0 Hz
        J = np.zeros((1, 1))  # No coupling for DSS
        return v, J

    return createcitrate, createdss, np


@app.cell
def _(createcitrate, createdss, np, spectrum_creator):
    import matplotlib.pyplot as plt
    # Create the analytes data
    v_dss, j_dss = createdss()
    v_citrate, j_citrate = createcitrate()

    basespectrum = spectrum_creator(
        analytes=[
            {
                'v': v_dss,
                'j': j_dss,
                'protons': 9,
                'concentration': 1.0
            },
            {
                'v': v_citrate,
                'j': j_citrate,
                'protons': 4,
                'concentration': 0.1
            }
        ]
    )

    print(np.array(basespectrum).shape)

    plt.title('Base spectrum')
    plt.plot(basespectrum[0], basespectrum[1])

    return j_citrate, j_dss, plt, v_citrate, v_dss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, create multiple of these spectra with varying concentrations""")
    return


@app.cell
def _(j_citrate, j_dss, np, spectrum_creator, v_citrate, v_dss):
    count = 100

    labels = []
    spectra = []

    for spectracounter in range (0, count):
        print(f'{spectracounter}/{count}')
        concentration = np.random.uniform(0.5, 0.01)
        spectra.append(spectrum_creator(
            analytes=[
                {
                    'v': v_dss,
                    'j': j_dss,
                    'protons': 9,
                    'concentration': 1.0
                },
                {
                    'v': v_citrate,
                    'j': j_citrate,
                    'protons': 4,
                    'concentration': concentration
                }
            ]
        ))

    return (spectra,)


@app.cell
def _(plt, spectra):
    print(len(spectra))

    graph_count = 9

    for graphcounter in range(graph_count):
        plt.subplot(1, int(graph_count/3), graphcounter)
        plt.plot(spectra[graphcounter][0], spectra[graphcounter][1])
    return


if __name__ == "__main__":
    app.run()
