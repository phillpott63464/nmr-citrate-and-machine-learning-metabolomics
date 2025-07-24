import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import time


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

    # Perform 3D adaptive grid refinement
    result = adaptive_grid_refinement_3d(bounds, known_values, search_molarity)

    pka = result

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


def evaluate_pka_error(pka_values, known_values, search_molarity):
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


def adaptive_grid_refinement_3d(
    bounds,
    known_values,
    search_molarity,
    depth=0,
    previous_best_error=float('inf'),
):
    """3D adaptive grid refinement for optimal pKa values"""

    # Generate a 3x3x3 grid of points within current bounds for more thorough search
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pka1 = round(
                    bounds[0][0] + (bounds[0][1] - bounds[0][0]) * i / 2, 2
                )
                pka2 = round(
                    bounds[1][0] + (bounds[1][1] - bounds[1][0]) * j / 2, 2
                )
                pka3 = round(
                    bounds[2][0] + (bounds[2][1] - bounds[2][0]) * k / 2, 2
                )
                points.append([pka1, pka2, pka3])

    # Remove duplicates
    unique_points = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)

    # Evaluate all points
    results = []
    for point in unique_points:
        error = evaluate_pka_error(point, known_values, search_molarity)
        results.append({'error': error, 'pkas': point})

    # Find the best point
    best = min(results, key=lambda x: x['error'])

    # Check if search space is too small to continue (primary stopping condition)
    ranges = [bounds[i][1] - bounds[i][0] for i in range(3)]
    if all(r < 0.01 for r in ranges):
        return best

    # Only stop if we haven't improved AND the search space is getting very small
    if best['error'] >= previous_best_error and all(r < 0.05 for r in ranges):
        return best

    # Create new bounds around the best point (smaller search area)
    best_point = best['pkas']
    new_bounds = []

    for i in range(3):
        current_range = bounds[i][1] - bounds[i][0]
        half_range = current_range / 3  # Smaller step for more precision
        new_min = max(bounds[i][0], best_point[i] - half_range)
        new_max = min(bounds[i][1], best_point[i] + half_range)
        new_bounds.append([new_min, new_max])

    # Recursively refine the grid in the refined space
    return adaptive_grid_refinement_3d(
        new_bounds, known_values, search_molarity, depth + 1, best['error']
    )


if __name__ == '__main__':
    main()
