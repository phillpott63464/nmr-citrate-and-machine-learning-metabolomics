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
    mo.md(r"""Outputs:""")
    return


@app.cell
def _(corrected_pka, fracs, phs, pka, plt):

    print('Calculated pkas:')
    print(pka)
    print('Corrected pkas:')
    print([round(x, 2) for x in corrected_pka])

    plt.plot(phs, fracs)
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    return


@app.cell
def _(mo):
    mo.md(r"""Define constants""")
    return


@app.cell
def _(electrolytes):
    EPS_R = 78.3   # relative permittivity of water at 25°C, should really change to 30C, maybe
    T = 303   # K
    RHO = 0.997
    B0 = 1.0
    A_CONST = electrolytes.A(eps_r=EPS_R, T=T, rho=RHO, b0=B0)
    trials = 10000
    search_molarity = 0.1
    graph_molarity = 0.001
    return A_CONST, graph_molarity, search_molarity, trials


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
            trial.suggest_float('pka1', low=2.0, high=3.5, step=0.01),
            trial.suggest_float('pka2', low=4.0, high=5.5, step=0.01),
            trial.suggest_float('pka3', low=5.5, high=6.5, step=0.01),
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
    return (known_values,)


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
        round(study.best_trial.params[x], 4) for x in study.best_trial.params
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
    return


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

    plt.plot(phs, fracs)
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    return fracs, phs, plt


if __name__ == '__main__':
    app.run()
