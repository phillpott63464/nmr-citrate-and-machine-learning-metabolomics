import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from functools import partial
from chempy import electrolytes

# Constants
EPS_R = 78.3   # relative permittivity of water at 25°C, should really change to 30C, maybe
T = 303   # K
RHO = 0.997
B0 = 1.0
A_CONST = electrolytes.A(eps_r=EPS_R, T=T, rho=RHO, b0=B0)
trials = 10000
search_molarity = 0.1
graph_molarity = 0.001


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


def main():

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
        n_jobs=8,
        callbacks=[
            optuna.study.MaxTrialsCallback(
                trials, states=(optuna.trial.TrialState.COMPLETE,)
            )
        ],
    )

    # Best pKa from 0.1 M calibration
    pka = [
        round(study.best_trial.params[x], 4) for x in study.best_trial.params
    ]

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

    with open('pka.txt', 'w') as f:
        f.write(f'Error: {round(study.best_trial.value, 10)}\n')
        f.write(f'Original pKas (0.1 M): {pka}\n')
        f.write(
            f'Corrected pKas (0.001 M): {[round(x, 2) for x in corrected_pka]}\n'
        )

    ratios = simulate_ph_graph(pka=corrected_pka, conc=graph_molarity)

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv('ratios.csv', index=False)

    phs = np.linspace(1, 9, 1000)

    citricacid = phfork.AcidAq(
        pKa=corrected_pka, charge=0, conc=graph_molarity
    )
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
