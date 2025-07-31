import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    import pandas as pd
    out_dir='experimental'


    imported = pd.read_csv(f'{out_dir}/eppendorfs.csv')
    imported = imported.to_dict(orient='split', index=False)

    eppendorfs = {
        imported['columns'][i]: {row[i] for row in imported['data']}
        for i in range(len(imported['columns']))
    }

    stocks = {
        'base': {
            'D2O': 2.5,
            'DSS': 22.64/2,
            'weight': 129.60,
            'water': 44.78,
        },
        'acid': {
            'D2O': 2.5,
            'DSS': 22.64/2,
            'weight': 95.95,
            'water': 40.37,
        }
    }

    return


if __name__ == "__main__":
    app.run()
