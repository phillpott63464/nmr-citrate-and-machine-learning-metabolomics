import marimo

__generated_with = '0.14.16'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""# Experimental Procedure Notes""")
    return


@app.cell
def _(mo, output, stocks):
    mo.md(
        rf"""
    ## Stocks:
    - Base concentration: {round(stocks['base']['molarity'], 5)}, target = 0.001
    - Acid concentration: {round(stocks['acid']['molarity'], 5)}, target = 0.001

    ## Total volume eppendorfs:
    {output}
    """
    )
    return


@app.cell
def _():
    import pandas as pd

    out_dir = 'experimental'
    acid_molecular_weight = 192.12   # g/mol
    base_molecular_weight = 258.07   # g/mol
    dss_molecular_weight = 224.36   # g/mol

    imported = pd.read_csv(f'{out_dir}/eppendorfs.csv')
    # imported = imported.to_dict(orient='split', index=False)

    acid_vol = [
        round((x - y) / 1000, 6)
        for x, y in zip(imported['acid'], imported['weight'])
    ]

    acid_vol[0] = 0.0006
    acid_vol[-1] = 0.0

    base_vol = [
        round((x - y) / 1000, 6)
        for x, y in zip(imported['base'], imported['acid'])
    ]

    base_vol[0] = 0.0
    base_vol[-1] = 0.0006

    total_vol = [round(x + y, 6) for x, y in zip(acid_vol, base_vol)]

    phs = [x for x in imported['ph']]

    stocks = {
        'base': {
            'D2O': 2.5,
            'DSSweight': 22.64 / 2,
            'weight': 129.60,
            'volume': 50,
            'water': 44.78,
            'molecular_weight': 258.07,
        },
        'acid': {
            'D2O': 2.5,
            'DSSweight': 22.64 / 2,
            'weight': 95.95,
            'volume': 50,
            'water': 40.37,
            'molecular_weight': 192.12,
        },
    }

    # Loop through each stock type (base and acid)
    for stock_type in stocks:
        stocks[stock_type]['molarity'] = (
            (stocks[stock_type]['weight'] / 1000)  # g
            / stocks[stock_type]['molecular_weight']  # g/mol
            / (stocks[stock_type]['volume'] / 1000)  # L
        )

    output = '\n\n'.join(
        [
            f"""Total Volume: {round(x * 1000, 2)} ml,
            pH: {y},
            Acid ratio={round(((z*1000)/0.6), 2)},
            BaseRatio = {round(((i*1000)/0.6), 2)}"""
            for idx, (x, y, z, i) in enumerate(
                zip(total_vol, phs, acid_vol, base_vol)
            )
        ]
    )

    print(acid_vol[4] * 1000)
    print(base_vol[4] * 1000)
    print(total_vol[4] * 1000)

    return output, stocks


if __name__ == '__main__':
    app.run()
