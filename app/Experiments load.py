import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


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
    import optuna
    import numpy as np
    from chempy import electrolytes

    study = optuna.create_study(
        direction='minimize',
        study_name='PKA_STUDY',
        storage='sqlite:///model_database/PKA_STUDY.db',
        load_if_exists=True,
    )

    pka = [
        round(study.best_trial.params[x], 5) for x in study.best_trial.params
    ]

    EPS_R = 78.3   # relative permittivity of water at 25°C, should really change to 30C, maybe
    T = 298  # K
    RHO = 0.997
    B0 = 1.0
    A_CONST = electrolytes.A(eps_r=EPS_R, T=T, rho=RHO, b0=B0)

    def ionic_strength_from_conc(conc):
        """
        Simplified ionic strength calculator for citrate + Na+ system,
        assuming Na+ balances citrate charge.
        Citrate species charges: H3A=0, H2A-=-1, HA2-=-2, A3-=-3
        We approximate total charge based on fully dissociated species distribution.
        """
        # Approximate ionic strength I = 0.5 * sum(ci * zi^2)
        # For simplicity, treat total citrate as fully dissociated into average charge species.
        # Here, we just approximate I = conc * average charge squared * factor
        # To be more accurate, you’d calculate actual speciation first.
        # For now, just return ionic strength proportional to conc.
        return conc * 0.1  # crude approximation; you can improve this!

    def debeye_huckel_log_gamma(z, I):
        """Davies equation log10 gamma"""
        sqrt_I = np.sqrt(I)
        return -A_CONST * z**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)

    def correct_pkas(pkas, I_old, I_new, charges):
        """Adjust each pKa from old ionic strength to new ionic strength."""
        corrected = []
        for pka, (z_acid, z_base) in zip(pkas, charges):
            log_gamma_old = debeye_huckel_log_gamma(
                z_base, I_old
            ) - debeye_huckel_log_gamma(z_acid, I_old)
            log_gamma_new = debeye_huckel_log_gamma(
                z_base, I_new
            ) - debeye_huckel_log_gamma(z_acid, I_new)
            delta_pka = log_gamma_new - log_gamma_old
            corrected.append(pka + delta_pka)
        return corrected

    # H3A (0) ⇌ H2A- (-1)
    # H2A- (-1) ⇌ HA2- (-2)
    # HA2- (-2) ⇌ A3- (-3)
    charge_pairs = [(0, -1), (-1, -2), (-2, -3)]

    search_molarity = 0.1
    graph_molarity = 0.001

    # Calculate ionic strengths (crudely approximated)
    I_old = ionic_strength_from_conc(search_molarity)
    I_new = ionic_strength_from_conc(graph_molarity)

    # Correct pKas for change in ionic strength
    corrected_pka = [
        round(float(x), 2) for x in correct_pkas(pka, I_old, I_new, charge_pairs)
    ]

    print(pka)
    print(corrected_pka)
    return (corrected_pka,)


@app.cell
def _(corrected_pka):
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

    import phfork


    print(base_vol)

    expected_phs = [
        (system := phfork.System(
            phfork.IonAq(charge=1, conc=(y * stocks['base']['molarity'] * 3)),
            phfork.AcidAq(pKa=corrected_pka, charge=0, conc=x * stocks['acid']['molarity'] + y * stocks['base']['molarity'])
        )).pHsolve() or system.pH  # Call pHsolve() and then access the pH attribute
        for x, y in zip(acid_vol, base_vol)
    ]

    def calculate_ratio(percentage_d2o):
        # Molar masses
        molar_mass_d2o = 20.028  # g/mol for D2O
        molar_mass_h2o = 18.016   # g/mol for H2O

        # Assuming 1 liter of solution (1000 g)
        total_mass = 1000  # g
        mass_d2o = (percentage_d2o / 100) * total_mass
        mass_h2o = total_mass - mass_d2o

        # Calculate moles
        moles_d2o = mass_d2o / molar_mass_d2o
        moles_h2o = mass_h2o / molar_mass_h2o

        # Calculate the ratio a
        a = moles_d2o / (moles_d2o + moles_h2o)
        return a

        # Calculate moles
        moles_d2o = mass_d2o / molar_mass_d2o
        moles_h2o = mass_h2o / molar_mass_h2o

        # Calculate the ratio a
        a = moles_d2o / (moles_d2o + moles_h2o)
        return a


    # for idx, x in enumerate(expected_phs):
    #     a = calculate_ratio(5)
    #     phAlter = 0.3139*a+0.0854*a**2
    #     expected_phs[idx] += phAlter

    print(expected_phs)

    output = '\n\n'.join(
        [

            f"""Experiment {idx+1}:
            Total Volume: {round(x * 1000, 2)} ml,
            pH: {y},
            Expected pH: {round(d, 2)},
            Acid ratio={round(((z*1000)/0.6), 2)},
            BaseRatio = {round(((i*1000)/0.6), 2)}"""
            for idx, (x, y, z, i, d) in enumerate(
                zip(total_vol, phs, acid_vol, base_vol, expected_phs)
            )
        ]
    )

    print(acid_vol[4] * 1000)
    print(base_vol[4] * 1000)
    print(total_vol[4] * 1000)

    return acid_vol, expected_phs, output, phs, stocks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(acid_vol, expected_phs, phs):
    import matplotlib.pyplot as plt

    plt.plot([(z*1000)/0.6 for z in acid_vol], phs, label='Experimental pHs')
    plt.plot([(z*1000)/0.6 for z in acid_vol], expected_phs, label='Expected pHs')
    # plt.plot([((z*1000)/0.6) for z in acid_vol])
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
