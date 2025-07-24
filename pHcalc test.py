import marimo

__generated_with = '0.14.13'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    from pHcalc import Acid, Inert, System

    return Acid, Inert, System, mo


app._unparsable_cell(
    r"""
    pHcalc the pypi package is broken, ignore
    """,
    name='_',
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Import data from bufferdata.csv and convert from pandas array to dict
    - Export shape of data and len of data
    """
    )
    return


@app.cell
def _():
    import pandas as pd

    imported = pd.read_csv('bufferdata.csv')
    imported = imported.to_dict(orient='split', index=False)

    known_values = []

    for data in imported['data']:
        known_values.append(
            {'ph': data[0], 'acid ratio': data[1], 'base ratio': data[2]}
        )

    print(known_values[0])
    print(len(known_values))
    return


app._unparsable_cell(
    r"""
    - Create a speciation plot of citrate at a specified molarity
    """,
    name='_',
)


@app.cell
def _(Acid, Inert, System, np):
    graph_molarity = 0.1
    import matplotlib.pyplot as plt

    ratios = []
    pka = [1, 2, 3]

    citricacid = Acid(pKa=pka, charge=0, conc=graph_molarity)
    for i in range(0, 201):
        na_molarity = graph_molarity * 3 * (i / 200)
        na = Inert(charge=1, conc=na_molarity)
        system = System(citricacid, na)
        system.pHsolve()

        ratios.append(
            {
                'pH': round(system.pH, 2),
                'acid ratio': 100 - i / 2,
                'base ratio': i / 2,
            }
        )

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    return


if __name__ == '__main__':
    app.run()
