import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    citrate_molarity = 0.1
    pkas = [3.13, 4.76, 6.40]
    citricacid = phfork.AcidAq(pKa=pkas, charge=0, conc=citrate_molarity)

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    # plt.show()

    plt.savefig('graph.png')

    pkas1 = [
        3.13,  # Common Organic Chemistry
        3.02,  # Paper
        2.96,  # Pubchem
        2.63,  # Pubchem
        2.87,  # Pubchem
        3.44,  # Pubchem
    ]

    pkas2 = [
        4.76,  # Common Organic Chemistry
        4.78,  # Paper
        4.38,  # Pubchem
        4.11,  # Pubchem
        4.35,  # Pubchem
        5.02,  # Pubchem
    ]

    pkas3 = [
        6.40,  # Common Organic Chemistry
        6.02,  # Paper
        5.68,  # Pubchem
        5.34,  # Pubchem
        5.68,  # Pubchem
        6.55,  # Pubchem
    ]

    for pka1 in pkas1:
        for pka2 in pkas2:
            for pka3 in pkas3:
                pkas.append([pka1, pka2, pka3])

    out = []

    for pka in pkas:
        citricacid = phfork.AcidAq(pKa=pka, charge=0, conc=citrate_molarity)
        ratios = []
        for i in range(0, 101):
            na_molarity = citrate_molarity * 3 * (i / 100)
            na = phfork.IonAq(charge=1, conc=na_molarity)
            system = phfork.System(citricacid, na)
            system.pHsolve()

            ratios.append(
                {
                    'pH': round(system.pH, 2),
                    'citric acid ratio': 100 - i,
                    'citrate ratio': i,
                }
            )

        out.append(
            {
                'ph4': min(ratios, key=lambda d: abs(d['pH'] - 4)),
                'pkas': pka,
            }
        )

    closest = min(out, key=lambda d: abs(d['ph4']['citric acid ratio'] - 59))

    pka = closest['pkas']

    ratios = []
    citricacid = phfork.AcidAq(pKa=pka, charge=0, conc=citrate_molarity)
    for i in range(0, 101):
        na_molarity = citrate_molarity * 3 * (i / 100)
        na = phfork.IonAq(charge=1, conc=na_molarity)
        system = phfork.System(citricacid, na)
        system.pHsolve()

        ratios.append(
            {
                'pH': round(system.pH, 2),
                'citric acid ratio': 100 - i,
                'citrate ratio': i,
            }
        )

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv('ratios.csv', index=False)


if __name__ == '__main__':
    main()
