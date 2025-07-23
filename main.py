import pandas


def main():
    sample_vol = 0.0006   # l
    sample_conc = 0.1   # M
    acid_mass = 21.01   # g/l
    base_mass = 29.41   # g/1
    rounding = 3

    experiments = pandas.read_csv(
        filepath_or_buffer='experiments.csv',
        names=['ph', 'acid ratio', 'base ratio'],
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

    print(len(out))

    # acid_vol = 0.01
    # base_vol = 0.01

    acid_weight = acid_mass * acid_vol * sample_conc
    base_weight = base_mass * base_vol * sample_conc

    acid_weight = acid_weight * 100   # Convert g to mg
    base_weight = base_weight * 100

    acid_vol = acid_vol * 1000   # Convert l to ml
    base_vol = base_vol * 1000

    print(f'Acid weight: {round(acid_weight, rounding)}mg')
    print(f'Acid volume: {round(acid_vol, rounding)}ml')
    print('\n')
    print(f'Base weight: {round(base_weight, rounding)}mg')
    print(f'Base volume: {round(base_vol, rounding)}ml')
    print('\n')

    out = pandas.DataFrame.from_dict(out)

    out.to_csv(path_or_buf='out.csv', index=False)


if __name__ == '__main__':
    main()
