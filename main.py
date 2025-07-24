import pandas
import math
from decimal import Decimal, ROUND_CEILING


def main():
    sample_vol = 0.0006   # l
    sample_conc = 0.1   # M
    acid_mass = 21.01   # g/l, 0.1M
    base_mass = 29.41   # g/1, 0.1M
    rounding = 3
    balance = '0.01'   # In quotations because reasons?

    ratios = pandas.read_csv(
        filepath_or_buffer='ratios.csv',
        names=['ph', 'acid ratio', 'base ratio'],
    )

    print(ratios.loc[ratios['ph'] - 4.abs().idxmin()])

    options = [4, 3, 2]

    print(min(ratios, key=lambda d: abs(d['pH'] - 4.0)))

    experiments = []
    for option in options:
        experiments.append(
            {
                'ph': min(ratios, key=lambda d: abs(d['pH'] - option))
            }
        )

    acid_vol = 0
    base_vol = 0

    out = []

    for index, row in experiments.iterrows():
        if index == 0:
            continue

        acid_multiplier = float(row['acid ratio']) / 100
        acid_vol_add = acid_multiplier * sample_vol

        base_multiplier = float(row['base ratio']) / 100
        base_vol_add = base_multiplier * sample_vol

        acid_vol += acid_vol_add
        base_vol += base_vol_add

        out.append(
            {
                'ph': row['ph'],
                'acid volume': round(acid_vol_add * 1000, rounding),
                'base volume': round(base_vol_add * 1000, rounding),
            }
        )

    acid_weight = acid_mass * acid_vol
    base_weight = base_mass * base_vol

    print(f'Actual requirements:')
    print(f'Acid weight: {round(acid_weight * 100, rounding)}mg')
    print(f'Acid volume: {round(acid_vol * 1000, rounding)}ml')
    print('\n')
    print(f'Base weight: {round(base_weight * 100, rounding)}mg')
    print(f'Base volume: {round(base_vol * 1000, rounding)}ml')
    print('\n')

    print('Requirements based upon the validity of a balance:')

    acid_weight = Decimal(acid_weight)
    acid_weight = acid_weight.quantize(
        Decimal(balance), rounding=ROUND_CEILING
    )

    base_weight = Decimal(base_weight)
    base_weight = base_weight.quantize(
        Decimal(balance), rounding=ROUND_CEILING
    )

    print(f'Acid weight: {round(float(acid_weight) * 100, rounding)}mg')
    print(
        f'Acid volume: {round(float(acid_weight)/acid_mass * 1000, rounding)}ml'
    )
    print('\n')
    print(f'Base weight: {round(float(base_weight) * 100, rounding)}mg')
    print(
        f'Base volume: {round(float(base_weight)/base_mass * 1000, rounding)}ml'
    )
    print('\n')

    out = pandas.DataFrame.from_dict(out)
    out.to_csv(path_or_buf='out.csv', index=False)


if __name__ == '__main__':
    main()
