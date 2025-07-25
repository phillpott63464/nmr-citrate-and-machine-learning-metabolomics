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

    # Define search bounds for pKa values
    bounds = [
        [2.0, 3.5],  # pKa1 range
        [4.0, 5.5],  # pKa2 range
        [5.5, 6.5],  # pKa3 range
    ]

    study = optuna.create_study(
        direction='minimize',
        study_name='PKA_STUDY',
        storage='sqlite:///PKA_STUDY.db',
        load_if_exists = True,
    )

    study.optimize(
        partial(objective, known_values=known_values, search_molarity=search_molarity),
          n_trials=10,
    )

    print(study.best_trial.params)


    with open('pka.txt', 'w') as f:
        f.write(str(pka))

    ratios = []
    citricacid = phfork.AcidAq(pKa=pka['pkas'], charge=0, conc=graph_molarity)
    for i in range(0, 201):
        na_molarity = graph_molarity * 3 * (i / 200)
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

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv('ratios.csv', index=False)

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.savefig('graph.png')


def evaluate_pka_error(known_values, search_molarity, trial):
    pka_values = [
        trial.suggest_float('pka1', low=2.0, high=3.5, step=0.01),
        trial.suggest_float('pka2', 4.0, 5.5, step=0.01),
        trial.suggest_float('pka3', 5.5, 6.5, step=0.01),
    ]

    print(pka_values)

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

    return error, pka_values

def objective(trial, search_molarity, known_values):
    error = evaluate_pka_error(trial=trial, search_molarity=search_molarity, known_values=known_values)
    return error

if __name__ == '__main__':
    main()
