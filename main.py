import pandas

def main():
    sample_vol = 0.0006 #l
    sample_conc = 0.1 #M
    acid_mass = 21.01 #g/l
    base_mass = 29.41 #g/1

    experiments = pandas.read_csv('experiments.csv', names=['ph', 'acid ratio', 'base ratio'])  

    acid_vol = 0
    base_vol = 0

    for index, row in experiments.iterrows():
        if index == 0:
            continue

        multiplier = float(row['acid ratio'])/100
        acid_vol += multiplier * sample_vol

        multiplier = float(row['base ratio'])/100
        base_vol += multiplier * sample_vol

    acid_weight = acid_mass*acid_vol*sample_conc
    base_weight = base_mass*base_vol*sample_conc

    print(f'Acid weight: {round(acid_weight*100, 2)}mg')
    print(f'Acid volume: {round(acid_vol*1000, 2)}ml')
    print('\n')
    print(f'Base weight: {round(base_weight*100, 2)}mg')
    print(f'Base volume: {round(base_vol*1000, 2)}ml')


    



if __name__ == '__main__':
    main()
