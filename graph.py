import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threading


def main():
    citrate_molarity = 0.1

    imported = pd.read_csv('bufferdata.csv')
    imported = imported.to_dict(orient='split', index=False)

    known_values = []

    for data in imported['data']:
        known_values.append(
            {'ph': data[0], 'acid ratio': data[1], 'base ratio': data[2]}
        )

    pkas = []

    for pka1 in range(200, 350, 10):
        for pka2 in range(400, 550, 10):
            for pka3 in range(550, 650, 10):
                pkas.append([pka1 / 100, pka2 / 100, pka3 / 100])

    out = []

    print(f'This many trials: {len(pkas)}')

    trial = 0
    for pka in pkas:
        trial += 1
        print(f'Trial: {trial}/{len(pkas)}')
        citricacid = phfork.AcidAq(pKa=pka, charge=0, conc=citrate_molarity)
        ratios = []
        error = 0.0
        for i in range(0, 101):
            na_molarity = citrate_molarity * 3 * (i / 100)
            na = phfork.IonAq(charge=1, conc=na_molarity)
            system = phfork.System(citricacid, na)
            system.pHsolve()

            ratios.append(
                {
                    'pH': round(system.pH, 2),
                    'acid ratio': 100 - i,
                    'base ratio': i,
                }
            )

        for ph in known_values:
            closest_ph = min(ratios, key=lambda d: abs(d['pH'] - ph['ph']))
            error += (ph['acid ratio'] - closest_ph['acid ratio']) ** 2

        out.append(
            {
                'error': error,
                'pkas': pka,
            }
        )   # The pH values closest to these values


    pka = min(out, key=lambda x: x['error'])

    print(pka)

    ratios = []
    citricacid = phfork.AcidAq(pKa=pka['pkas'], charge=0, conc=citrate_molarity)
    for i in range(0, 101):
        na_molarity = citrate_molarity * 3 * (i / 100)
        na = phfork.IonAq(charge=1, conc=na_molarity)
        system = phfork.System(citricacid, na)
        system.pHsolve()

        ratios.append(
            {
                'pH': round(system.pH, 2),
                'acid ratio': 100 - i,
                'base ratio': i,
            }
        )

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv('ratios.csv', index=False)

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.savefig('graph.png')


if __name__ == '__main__':
    main()
