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
        round(float(x), 2)
        for x in correct_pkas(pka, I_old, I_new, charge_pairs)
    ]

    print(pka)
    print(corrected_pka)
    return corrected_pka, graph_molarity, np, pka


@app.cell
def _(np, pka):
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

    def simulate_ph_graph(pka, conc, charge=0):
        ratios = []
        citricacid = phfork.AcidAq(pKa=pka, charge=0, conc=conc)
        for i in range(0, 201):
            na_molarity = conc * 3 * (i / 200)
            na = phfork.IonAq(charge=1, conc=na_molarity)
            system = phfork.System(citricacid, na)
            system.pHsolve()

            ratios.append(
                {
                    'pH': round(system.pH, 2),
                    'acid ratio': 100 - i / 2,
                    'base ratio': i / 2,
                }
            )

        return ratios

    ratios = simulate_ph_graph(
        pka, ((stocks['acid']['molarity'] + stocks['base']['molarity']) / 2)
    )

    all_expected_phs = [x['pH'] for x in ratios]
    acid_ratios = [x['acid ratio'] / 100 for x in ratios]
    acid_experimental_ratios = [(z * 1000) / 0.6 for z in acid_vol]
    # acid_experimental_ratios = [z for z in acid_vol]

    print(acid_ratios)
    print(acid_experimental_ratios)

    closest_indices = []
    for exp_ratio in acid_experimental_ratios:
        # Calculate the absolute differences
        differences = np.abs(np.array(acid_ratios) - exp_ratio)
        # Find the index of the minimum difference
        closest_index = np.argmin(differences)
        closest_indices.append(closest_index)

    expected_phs = []
    for idx in closest_indices:
        expected_phs.append(all_expected_phs[idx])

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

    for idx, x in enumerate(expected_phs):
        a = calculate_ratio(5)
        phAlter = 0.3139 * a + 0.0854 * a**2
        expected_phs[idx] += phAlter

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

    return (
        acid_vol,
        base_vol,
        expected_phs,
        output,
        phs,
        simulate_ph_graph,
        stocks,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(acid_vol, corrected_pka, expected_phs, phs):
    def _():
        import matplotlib.pyplot as plt

        plt.plot(
            [(z * 1000) / 0.6 for z in acid_vol], phs, label='Experimental pHs'
        )
        plt.plot(
            [(z * 1000) / 0.6 for z in acid_vol],
            expected_phs,
            label='Expected pHs',
        )
        for id, point in enumerate(corrected_pka):
            plt.axhline(
                y=point,
                color='red',
                linestyle='--',
                label=f'pka{id+1} = {point}',
            )
        # plt.plot([((z*1000)/0.6) for z in acid_vol])
        return plt.legend()

    _()
    return


@app.cell
def _(acid_vol, base_vol, corrected_pka, expected_phs, phs, stocks):
    import matplotlib.pyplot as plt

    # Assuming acid_vol and base_vol are already in liters
    # Ensure you have the correct molarity for both acid and base
    moles_acid = [vol * stocks['acid']['molarity'] for vol in acid_vol]
    moles_base = [
        vol * stocks['base']['molarity'] for vol in base_vol
    ]  # Use base molarity here

    # Calculate molar ratios
    # molar_ratios = [acid / base if base != 0 else 7 for acid, base in zip(moles_acid, moles_base)]
    molar_ratios = [
        (base - acid) for acid, base in zip(moles_acid, moles_base)
    ]

    def _():
        dels = [5, 6, 16]
        for i in dels:

            del molar_ratios[i]
            del phs[i]
            del expected_phs[i]

    # _()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(molar_ratios, phs, label='Experimental pHs', marker='o')
    plt.plot(molar_ratios, expected_phs, label='Expected pHs', marker='x')
    # plt.plot([molar_ratios[0], molar_ratios[-1]], [phs[0], phs[-1]], label='Linear phs')

    for id, point in enumerate(corrected_pka):
        plt.axhline(
            y=point, color='red', linestyle='--', label=f'pka{id+1} = {point}'
        )

    plt.title('Effect of Molar Ratio on pH Values')
    plt.xlabel('Molar Ratio (normalized)')
    # plt.xscale('log') #Problematic line
    plt.ylabel('pH Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

    return (plt,)


@app.cell
def _(corrected_pka, graph_molarity, plt, simulate_ph_graph):
    ratios_perfect = simulate_ph_graph(corrected_pka, graph_molarity, charge=0)

    print(ratios_perfect)

    moles_acid_perfect = [
        ratio * 0.006 * graph_molarity
        for ratio in [x['acid ratio'] for x in ratios_perfect]
    ]
    moles_base_perfect = [
        ratio * 0.006 * graph_molarity
        for ratio in [x['base ratio'] for x in ratios_perfect]
    ]

    molar_ratios_perfect = [
        (base - acid)
        for acid, base in zip(moles_acid_perfect, moles_base_perfect)
    ]

    plt.figure(figsize=(10, 6))

    # plt.plot([x['acid ratio'] for x in ratios_perfect], [x['pH'] for x in ratios_perfect])
    plt.plot(molar_ratios_perfect, [x['pH'] for x in ratios_perfect])

    def _():
        for id, point in enumerate(corrected_pka):
            plt.axhline(
                y=point,
                color='red',
                linestyle='--',
                label=f'pka{id+1} = {point}',
            )

    _()

    plt.title('Effect of Molar Ratio on pH Values (Perfect values)')
    plt.xlabel('Molar Ratio')
    # plt.xscale('log') #Problematic line
    plt.ylabel('pH Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

    return


if __name__ == '__main__':
    app.run()
