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

    system = SpinSystem(vinyl[0], vinyl[1])

    system.peaklist()
    return SpinSystem, system


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot using nmrsim""")
    return


@app.cell
def _(system):
    from nmrsim.plt import mplplot

    mplplot(system.peaklist(), y_max=0.2)
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
    citratesystem = SpinSystem(citrate[0], citrate[1])

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
def _(citratesystem, mplplot, system):
    from nmrsim import Spectrum

    mix = Spectrum([citratesystem, system])
    mplplot(mix.peaklist())
    return


if __name__ == "__main__":
    app.run()
