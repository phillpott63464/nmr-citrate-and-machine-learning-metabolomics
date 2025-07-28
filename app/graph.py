import marimo

__generated_with = '0.14.13'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    from phfork import AcidAq, IonAq, System
    import phfork
    from chempy import electrolytes
    import numpy as np

    return electrolytes, mo, np, phfork


@app.cell
def _(mo):
    mo.md(r"""Define constants""")
    return


@app.cell
def _(electrolytes):
    # Graph constants
    EPS_R = 78.3   # relative permittivity of water at 25°C, should really change to 30C, maybe
    T = 303  # K
    RHO = 0.997
    B0 = 1.0
    A_CONST = electrolytes.A(eps_r=EPS_R, T=T, rho=RHO, b0=B0)
    trials = 10
    search_molarity = 0.1
    graph_molarity = 0.001
    stock_molarity = 0.01
    sample_vol = 0.0006   # l
    acid_mass = 21.01   # g/l, 0.1M
    base_mass = 29.41   # g/1, 0.1M
    rounding = 3
    balance = '0.1'   # In quotations because reasons?

    options = [
        2.1,
        3.2,
        3.5,
        3.7,  # 4
        3.8,
        4,
        4.2,
        4.4,  # 8
        4.5,
        4.6,
        4.8,
        5,  # 12
        5.2,
        5.4,
        5.5,
        5.7,  # 16
        5.9,
        6,
        6.2,
        6.4,  # 20
        6.6,
        7,
        7.4,
        8,  # 24
    ]
    return (
        A_CONST,
        acid_mass,
        balance,
        base_mass,
        graph_molarity,
        options,
        rounding,
        sample_vol,
        search_molarity,
        stock_molarity,
        trials,
    )


@app.cell
def _(mo):
    mo.md(
        r"""Define a function to simulate a ph graph at a concentration from pkas"""
    )
    return


@app.cell
def _(phfork):
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

    return (simulate_ph_graph,)


@app.cell
def _(mo):
    mo.md(
        r"""Define a function that will evaluate the mean square error in pH values between the known buffer data and the predicted data from a set of pkas"""
    )
    return


@app.cell
def _(simulate_ph_graph):
    def evaluate_pka_error(known_values, search_molarity, trial):
        pka_values = [
            trial.suggest_float('pka1', low=2.0, high=3.5, step=0.001),
            trial.suggest_float('pka2', low=4.0, high=5.5, step=0.001),
            trial.suggest_float('pka3', low=5.5, high=6.5, step=0.001),
        ]

        ratios = simulate_ph_graph(pka=pka_values, conc=search_molarity)

        error = 0.0
        for known in known_values:
            closest_entry = min(
                ratios,
                key=lambda d: abs(d['acid ratio'] - known['acid ratio']),
            )
            error += (known['ph'] - closest_entry['pH']) ** 2

        return error

    return (evaluate_pka_error,)


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
    return known_values, pd


@app.cell
def _(mo):
    mo.md(r"""- Create an optuna study and trial it""")
    return


@app.cell
def _(evaluate_pka_error, known_values, search_molarity, trials):
    import optuna
    from functools import partial

    def objective(trial, search_molarity, known_values):
        error = evaluate_pka_error(
            trial=trial,
            search_molarity=search_molarity,
            known_values=known_values,
        )
        return error

    study = optuna.create_study(
        direction='minimize',
        study_name='PKA_STUDY',
        storage='sqlite:///PKA_STUDY.db',
        load_if_exists=True,
    )

    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )

    if trials - completed_trials > 0:
        study.optimize(
            partial(
                objective,
                known_values=known_values,
                search_molarity=search_molarity,
            ),
            n_jobs=8,
            callbacks=[
                optuna.study.MaxTrialsCallback(
                    trials, states=(optuna.trial.TrialState.COMPLETE,)
                )
            ],
        )

    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Get pKas of trial data""")
    return


@app.cell
def _(study):
    pka = [
        round(study.best_trial.params[x], 5) for x in study.best_trial.params
    ]
    print(pka)
    return (pka,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define functions for debye huckel correction""")
    return


@app.cell
def _(A_CONST, np):
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

    return correct_pkas, ionic_strength_from_conc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Correct for ion concentration using debye huckel""")
    return


@app.cell
def _(
    correct_pkas,
    graph_molarity,
    ionic_strength_from_conc,
    pka,
    search_molarity,
):
    # Citrate acid-base pairs and charges for pKa steps:
    # H3A (0) ⇌ H2A- (-1)
    # H2A- (-1) ⇌ HA2- (-2)
    # HA2- (-2) ⇌ A3- (-3)
    charge_pairs = [(0, -1), (-1, -2), (-2, -3)]

    # Calculate ionic strengths (crudely approximated)
    I_old = ionic_strength_from_conc(search_molarity)
    I_new = ionic_strength_from_conc(graph_molarity)

    # Correct pKas for change in ionic strength
    corrected_pka = [
        float(x) for x in correct_pkas(pka, I_old, I_new, charge_pairs)
    ]

    print(pka)
    print([round(x, 2) for x in corrected_pka])
    return (corrected_pka,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create final pH to ratio dataframe""")
    return


@app.cell
def _(corrected_pka, graph_molarity, simulate_ph_graph):
    ratios = simulate_ph_graph(pka=corrected_pka, conc=graph_molarity)
    return (ratios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create final speciation graph""")
    return


@app.cell
def _(corrected_pka, graph_molarity, np, phfork):
    import matplotlib.pyplot as plt

    phs = np.linspace(1, 9, 1000)

    citricacid = phfork.AcidAq(
        pKa=corrected_pka, charge=0, conc=graph_molarity
    )
    fracs = citricacid.alpha(phs)

    fig = plt.figure()

    plt.plot(phs, fracs)
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Choose options from calculated ratios""")
    return


@app.cell
def _(options, ratios):
    experiments = []
    for option in options:
        experiments.append(
            min(ratios, key=lambda d: abs(float(d['pH']) - option))
        )

    print(''.join(f'{x}\n' for x in experiments))

    return (experiments,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Calculate volume of acid/base needed to complete these experiments"""
    )
    return


@app.cell
def _(experiments, rounding, sample_vol, stock_molarity):
    acid_vol = 0
    base_vol = 0

    volumed_experiments = []

    for row in experiments:

        acid_multiplier = float(row['acid ratio']) / 100
        acid_vol_add = acid_multiplier * sample_vol

        base_multiplier = float(row['base ratio']) / 100
        base_vol_add = base_multiplier * sample_vol

        acid_vol += acid_vol_add
        base_vol += base_vol_add

        volumed_experiments.append(
            {
                'pH': row['pH'],
                'acid volume': round(
                    acid_vol_add * 1 / stock_molarity, rounding
                ),
                'base volume': round(
                    base_vol_add * 1 / stock_molarity, rounding
                ),
            }
        )

    print(''.join(f'{x}\n' for x in volumed_experiments))
    return acid_vol, base_vol, volumed_experiments


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define stock requirements""")
    return


@app.cell
def _(acid_mass, acid_vol, base_mass, base_vol, rounding):
    stock_output = []

    def stock(msg):
        stock_output.append(f'{msg}\n')
        print(msg)

    acid_weight = acid_mass * acid_vol
    base_weight = base_mass * base_vol

    stock(f'Actual requirements:')
    stock(f'Acid weight: {round(acid_weight * 100, rounding)}mg')
    stock(f'Acid volume: {round(acid_vol * 1000, rounding)}ml')
    stock('\n')
    stock(f'Base weight: {round(base_weight * 100, rounding)}mg')
    stock(f'Base volume: {round(base_vol * 1000, rounding)}ml')
    return acid_weight, base_weight, stock, stock_output


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define stock requirements based upon the validty of a balance""")
    return


@app.cell
def _(
    acid_mass,
    acid_weight,
    balance,
    base_mass,
    base_weight,
    rounding,
    stock,
    stock_molarity,
):
    from decimal import Decimal, ROUND_CEILING

    acid_weight_balance = Decimal(acid_weight).quantize(
        Decimal(balance), rounding=ROUND_CEILING
    )
    base_weight_balance = Decimal(base_weight).quantize(
        Decimal(balance), rounding=ROUND_CEILING
    )

    stock(f'\nRequirements for {stock_molarity}M stock solution:')
    stock(
        f'Acid weight: {round(float(acid_weight_balance) * 100, rounding)}mg'
    )
    stock(
        f'Acid volume: {round(float(acid_weight_balance)/acid_mass * 1/stock_molarity, rounding)}ml'
    )
    stock('\n')
    stock(
        f'Base weight: {round(float(base_weight_balance) * 100, rounding)}mg'
    )
    stock(
        f'Base volume: {round(float(base_weight_balance)/base_mass * 1/stock_molarity, rounding)}ml'
    )

    stupid_variable = True   # Required otherwise cell 30 will run before cell 28 and thus cut off the results for some reason
    return (stupid_variable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Save everything important to files""")
    return


@app.cell
def _(
    corrected_pka,
    fig,
    pd,
    pka,
    stock_output,
    study,
    stupid_variable,
    volumed_experiments,
):
    import operator
    import os

    directory = 'output-graph'

    if (
        stupid_variable == False
    ):   # Required otherwise cell 30 will run before cell 28 and thus cut off the results for some reason
        print('Done a stupid')

    if not os.path.isdir(directory):
        os.mkdir(directory)

    with open(f'{directory}/pka.txt', 'w') as f:
        f.write(f'Error: {round(study.best_trial.value, 10)}\n')
        f.write(f'Original pKas (0.1 M): {pka}\n')
        f.write(
            f'Corrected pKas (0.001 M): {[round(x, 3) for x in corrected_pka]}\n'
        )

    with open(file=f'{directory}/stocks.txt', mode='w') as f:
        f.writelines(stock_output)

    volumed_experiments.sort(key=operator.itemgetter('pH'))

    out = pd.DataFrame.from_dict(volumed_experiments)
    out.to_csv(path_or_buf=f'{directory}/experiments.csv', index=False)

    fig.savefig(f'{directory}/graph.png')
    return


if __name__ == '__main__':
    app.run()
