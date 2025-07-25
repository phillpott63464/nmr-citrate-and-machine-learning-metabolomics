import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium", layout_file="layouts/pHcalc test.slides.json")


@app.cell
def _():
    import marimo as mo
    from phfork import AcidAq, IonAq, System
    import phfork

    return AcidAq, IonAq, System, mo, phfork


@app.cell
def _(mo):
    mo.md(r"""pHcalc the pypi package is broken, ignore""")
    return


@app.cell
def _(mo):
    mo.md(r"""- Define a function that will evaluate the mean square error in pH values between the known buffer data and the predicted data from a set of pkas""")
    return


@app.cell
def _(phfork):
    def evaluate_pka_error(known_values, search_molarity, trial):
        pka_values = [
            trial.suggest_float('pka1', low=2.0, high=3.5, step=0.01),
            trial.suggest_float('pka2', low=4.0, high=5.5, step=0.01),
            trial.suggest_float('pka3', low=5.5, high=6.5, step=0.01),
        ]

        """Evaluate error for a single pKa combination"""
        citricacid = phfork.AcidAq(pKa=pka_values, charge=0, conc=search_molarity)
        ratios = []

        for i in range(0, 201):
            na_molarity = search_molarity * 3 * (i / 200)
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

        error = 0.0
        for ph in known_values:
            closest_ph = min(ratios, key=lambda d: abs(d['pH'] - ph['ph']))
            error += (ph['acid ratio'] - closest_ph['acid ratio']) ** 2

        return error
    return (evaluate_pka_error,)


@app.cell
def _(mo):
    mo.md(r"""- Create an optuna study and trial it""")
    return


@app.cell
def _(evaluate_pka_error, known_values):
    import optuna
    from functools import partial

    search_molarity=0.1

    def objective(trial, search_molarity, known_values):
        error = evaluate_pka_error(
            trial=trial, search_molarity=search_molarity, known_values=known_values
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
        n_trials=10,
        n_jobs=8,
    )
    return (study,)


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
    mo.md(r"""- Create a speciation plot of citrate at a specified molarity""")
    return


@app.cell
def _(AcidAq, IonAq, System, study):
    graph_molarity = 0.001

    import matplotlib.pyplot as plt
    import numpy as np

    ratios = []
    pka = [
        study.best_trial.params['pka1'],
        study.best_trial.params['pka2'],
        study.best_trial.params['pka3'],
    ]

    citricacid = AcidAq(pKa=pka, charge=0, conc=graph_molarity)
    for i in range(0, 201):
        na_molarity = graph_molarity * 3 * (i / 200)
        na = IonAq(charge=1, conc=na_molarity)
        system = System(citricacid, na)
        system.pHsolve()

        ratios.append(
            {
                'pH': round(system.pH, 2),
                'AcidAq ratio': 100 - i / 2,
                'base ratio': i / 2,
            }
        )

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    return


if __name__ == "__main__":
    app.run()
