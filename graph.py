import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from functools import partial


def main():
    search_molarity = 0.1
    graph_molarity = 0.001

    imported = pd.read_csv('bufferdata.csv')
    imported = imported.to_dict(orient='split', index=False)

    known_values = []

    for data in imported['data']:
        known_values.append(
            {'ph': data[0], 'acid ratio': data[1], 'base ratio': data[2]}
        )

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
        n_jobs = 8,
        callbacks=[
            optuna.study.MaxTrialsCallback(
                10000, states=(optuna.trial.TrialState.COMPLETE,)
            )
        ],
    )

    pka = [
        round(study.best_trial.params[x], 2) for x in study.best_trial.params
    ]

    with open('pka.txt', 'w') as f:
        f.write(f'Error: {round(study.best_trial.value, 10)}\npkas: {pka}')

    ratios = simulate_ph_graph(pka=pka, conc=graph_molarity)

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv('ratios.csv', index=False)

    phs = np.linspace(1, 9, 1000)

    citricacid = phfork.AcidAq(pKa=pka, charge=0, conc=graph_molarity)
    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.savefig('graph.png')


def evaluate_pka_error(known_values, search_molarity, trial):
    pka_values = [
        trial.suggest_float('pka1', low=2.0, high=3.5, step=0.01),
        trial.suggest_float('pka2', low=4.0, high=5.5, step=0.01),
        trial.suggest_float('pka3', low=5.5, high=6.5, step=0.01),
    ]

    """Evaluate error for a single pKa combination"""

    ratios = simulate_ph_graph(pka=pka_values, conc=search_molarity)

    error = 0.0
    for known in known_values:
        closest_entry = min(
            ratios, key=lambda d: abs(d['acid ratio'] - known['acid ratio'])
        )
        error += (known['ph'] - closest_entry['pH']) ** 2

    return error


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


def objective(trial, search_molarity, known_values):
    error = evaluate_pka_error(
        trial=trial, search_molarity=search_molarity, known_values=known_values
    )
    return error


if __name__ == '__main__':
    main()
