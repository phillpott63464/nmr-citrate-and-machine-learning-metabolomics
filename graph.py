import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    citrate_molarity = 0.001
    pkas = [3.13, 4.76, 6.40]
    citricacid = phfork.AcidAq(pKa=pkas, charge=0, conc=citrate_molarity)

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['CH3', 'CH2^-', 'CH^2-', 'C^3-'])

    # plt.show()

    plt.savefig('graph.png')

    ratios = []

    for i in range(0, 101):
        na_molarity = citrate_molarity * 3 * (i / 100)
        na = phfork.IonAq(charge=1, conc=na_molarity)
        system = phfork.System(citricacid, na)
        system.pHsolve()

        ratios.append(
            {
                'pH': round(system.pH, 2),
                'citrate ratio': i,
                'citric acid ratio': 100 - i,
            }
        )

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv(path_or_buf='ratios.csv', index=False)


if __name__ == '__main__':
    main()
