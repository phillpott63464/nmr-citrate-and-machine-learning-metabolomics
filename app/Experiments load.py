import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""# Experimental Procedure Notes""")
    return


@app.cell
def _(graph_molarity, mo, output, stocks):
    mo.md(
        rf"""
    ## Stocks:
    - Base concentration: {round(stocks['base']['molarity'], 5)}, target = {graph_molarity}
    - Acid concentration: {round(stocks['acid']['molarity'], 5)}, target = {graph_molarity}

    ## Total volume eppendorfs:
    {output}
    """
    )
    return


@app.cell
def _():
    import optuna
    import numpy as np
    from chempy import electrolytes

    study = optuna.create_study(
        direction='minimize',
        study_name='PKA_STUDY',
        storage='sqlite:///PKA_STUDY.db',
        load_if_exists=True,
    )

    pka = [
        round(study.best_trial.params[x], 5) for x in study.best_trial.params
    ]

    EPS_R = 78.3   # relative permittivity of water at 25°C, should really change to 30C, maybe
    T = 298  # K
    RHO = 0.997
    B0 = 1.0
    A_CONST = electrolytes.A(eps_r=EPS_R, T=T, rho=RHO, b0=B0)

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

    # H3A (0) ⇌ H2A- (-1)
    # H2A- (-1) ⇌ HA2- (-2)
    # HA2- (-2) ⇌ A3- (-3)
    charge_pairs = [(0, -1), (-1, -2), (-2, -3)]

    search_molarity = 0.1
    graph_molarity = 0.01

    # Calculate ionic strengths (crudely approximated)
    I_old = ionic_strength_from_conc(search_molarity)
    I_new = ionic_strength_from_conc(graph_molarity)

    # Correct pKas for change in ionic strength
    corrected_pka = [
        round(float(x), 2)
        for x in correct_pkas(pka, I_old, I_new, charge_pairs)
    ]

    print(pka)
    print(corrected_pka)

    pkasolver = [2.95, 3.43, 3.98]
    return corrected_pka, graph_molarity, np, pkasolver


@app.cell
def _(corrected_pka, np, pkasolver):
    import pandas as pd

    out_dir = 'experimental'
    acid_molecular_weight = 192.12   # g/mol
    base_molecular_weight = 258.07   # g/mol
    dss_molecular_weight = 224.36   # g/mol

    imported = pd.read_csv(f'{out_dir}/eppendorfs.csv')
    # imported = imported.to_dict(orient='split', index=False)

    acid_vol = [
        round((x - y) / 1000, 6)
        for x, y in zip(imported['acid'], imported['weight'])
    ]

    acid_vol[0] = 0.0006
    acid_vol[-1] = 0.0

    base_vol = [
        round((x - y) / 1000, 6)
        for x, y in zip(imported['base'], imported['acid'])
    ]

    base_vol[0] = 0.0
    base_vol[-1] = 0.0006

    total_vol = [round(x + y, 6) for x, y in zip(acid_vol, base_vol)]

    phs = [x for x in imported['ph']]

    stocks = {
        'base': {
            'D2O': 2.5,
            'DSSweight': 22.64 / 2,
            'weight': 129.60,
            'volume': 50,
            'water': 44.78,
            'molecular_weight': 258.07,
        },
        'acid': {
            'D2O': 2.5,
            'DSSweight': 22.64 / 2,
            'weight': 95.95,
            'volume': 50,
            'water': 40.37,
            'molecular_weight': 192.12,
        },
    }

    # Loop through each stock type (base and acid)
    for stock_type in stocks:
        stocks[stock_type]['molarity'] = (
            (stocks[stock_type]['weight'] / 1000)  # g
            / stocks[stock_type]['molecular_weight']  # g/mol
            / (stocks[stock_type]['volume'] / 1000)  # L
        )

    import phfork

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

    ratios = simulate_ph_graph(
        corrected_pka, ((stocks['acid']['molarity'] + stocks['base']['molarity']) / 2)
    )

    pkasolver_ratios = simulate_ph_graph(
        pkasolver,  ((stocks['acid']['molarity'] + stocks['base']['molarity']) / 2)
    )

    all_expected_phs = [x['pH'] for x in ratios]
    all_pkasolver_phs = [x['pH'] for x in pkasolver_ratios]
    acid_ratios = [x['acid ratio'] / 100 for x in ratios]
    acid_experimental_ratios = [(z * 1000) / 0.6 for z in acid_vol]
    # acid_experimental_ratios = [z for z in acid_vol]

    closest_indices = []
    for exp_ratio in acid_experimental_ratios:
        # Calculate the absolute differences
        differences = np.abs(np.array(acid_ratios) - exp_ratio)
        # Find the index of the minimum difference
        closest_index = np.argmin(differences)
        closest_indices.append(closest_index)

    expected_phs = []
    pkasolver_phs = []
    expected_acid_ratios = []
    for idx in closest_indices:
        pkasolver_phs.append(all_pkasolver_phs[idx])
        expected_phs.append(all_expected_phs[idx])
        expected_acid_ratios.append(acid_ratios[idx])

    def calculate_ratio(percentage_d2o):
        # Molar masses
        molar_mass_d2o = 20.028  # g/mol for D2O
        molar_mass_h2o = 18.016   # g/mol for H2O

        # Assuming 1 liter of solution (1000 g)
        total_mass = 1000  # g[2.98, 4.385, 5.72]

        mass_d2o = (percentage_d2o / 100) * total_mass
        mass_h2o = total_mass - mass_d2o

        # Calculate moles
        moles_d2o = mass_d2o / molar_mass_d2o
        moles_h2o = mass_h2o / molar_mass_h2o

        # Calculate the ratio a
        a = moles_d2o / (moles_d2o + moles_h2o)
        return a

        # Calculate moles
        moles_d2o = mass_d2o / molar_mass_d2o
        moles_h2o = mass_h2o / molar_mass_h2o

        # Calculate the ratio a
        a = moles_d2o / (moles_d2o + moles_h2o)
        return a

    a = calculate_ratio(5) #5% D2O
    phAlter = 0.3139 * a + 0.0854 * a**2

    for idx, x in enumerate(expected_phs):
        expected_phs[idx] += phAlter

    for idx, x in enumerate(pkasolver_phs):
        pkasolver_phs[idx] += phAlter

    output = '\n\n'.join(
        [
            f"""Experiment {idx+1}:
            Total Volume: {round(x * 1000, 2)} ml,
            pH: {y},
            Expected pH: {round(d, 2)},
            Acid ratio={round(((z*1000)/0.6), 2)},
            BaseRatio = {round(((i*1000)/0.6), 2)}"""
            for idx, (x, y, z, i, d) in enumerate(
                zip(total_vol, phs, acid_vol, base_vol, expected_phs)
            )
        ]
    )

    return (
        acid_vol,
        base_vol,
        expected_acid_ratios,
        expected_phs,
        output,
        phfork,
        phs,
        pkasolver_phs,
        pkasolver_ratios,
        simulate_ph_graph,
        stocks,
    )


@app.cell(hide_code=True)
def _(mo, phgraph):
    mo.md(
        rf"""
    ## Measured pHs of Samples

    - Experimental: from pH meter
    - Expected: calculated from [phfork](https://github.com/mhvwerts/pHfork) using pKa values solved from [graph.py](?file=graph.py)

    {mo.as_html(phgraph)}
    """
    )
    return


@app.cell
def _(
    acid_vol,
    base_vol,
    corrected_pka,
    expected_acid_ratios,
    expected_phs,
    phs,
    pkasolver_phs,
    stocks,
):
    import matplotlib.pyplot as plt

    # Assuming acid_vol and base_vol are already in liters
    # Ensure you have the correct molarity for both acid and base
    moles_acid = [vol * stocks['acid']['molarity'] for vol in acid_vol]
    moles_base = [
        vol * stocks['base']['molarity'] for vol in base_vol
    ]  # Use base molarity here
    expected_moles_acid = [ratio * 0.0006 * stocks['acid']['molarity'] for ratio in expected_acid_ratios]
    expected_moles_base = [(1-ratio) * 0.0006 * stocks['acid']['molarity'] for ratio in expected_acid_ratios]

    # Calculate molar ratios
    # molar_ratios = [acid / base if base != 0 else 7 for acid, base in zip(moles_acid, moles_base)]
    molar_ratios = [
        (base - acid) for acid, base in zip(moles_acid, moles_base)
    ]

    expected_molar_ratios = [
        (base - acid) for acid, base in zip(expected_moles_acid, expected_moles_base)
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(molar_ratios, phs, label='Experimental pHs', marker='o')
    plt.plot(expected_molar_ratios, expected_phs, label='Expected pHs', marker='x')
    plt.plot(expected_molar_ratios, pkasolver_phs, label='Pkasolver pHs', marker='x')

    for id, point in enumerate(corrected_pka):
        plt.axhline(
            y=point, color='red', linestyle='--', label=f'pka{id+1} = {point}'
        )

    plt.title('Effect of Molar Ratio on pH Values')
    plt.xlabel('Molar Ratio')
    # plt.xscale('log') #Problematic line
    plt.ylabel('pH Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping

    phgraph = plt.gca()
    return phgraph, plt


@app.cell
def _(
    corrected_pka,
    graph_molarity,
    pkasolver,
    pkasolver_ratios,
    plt,
    simulate_ph_graph,
):
    ratios_perfect = simulate_ph_graph(corrected_pka, graph_molarity, charge=0)

    moles_acid_perfect = [
        ratio * 0.006 * graph_molarity
        for ratio in [x['acid ratio'] for x in ratios_perfect]
    ]
    moles_base_perfect = [
        ratio * 0.006 * graph_molarity
        for ratio in [x['base ratio'] for x in ratios_perfect]
    ]

    molar_ratios_perfect = [
        (base - acid)
        for acid, base in zip(moles_acid_perfect, moles_base_perfect)
    ]

    plt.figure(figsize=(10, 6))

    # plt.plot([x['acid ratio'] for x in ratios_perfect], [x['pH'] for x in ratios_perfect])
    plt.plot(molar_ratios_perfect, [x['pH'] for x in ratios_perfect], label='Solved pHs')
    # plt.plot(moles_acid_perfect, [x['pH'] for x in ratios_perfect])

    plt.plot(molar_ratios_perfect, [x['pH'] for x in pkasolver_ratios], label='Pkasolver pHs')

    def _():
        for id, point in enumerate(corrected_pka):
            plt.axhline(
                y=point,
                color='red',
                linestyle='--',
                label=f'pka{id+1} = {point}',
            )
        for id, point in enumerate(pkasolver):
            plt.axhline(
                y=point,
                color='blue',
                linestyle='--',
                label=f'pkasolver{id+1} = {point}',
            )

    # _()

    plt.title('Effect of Molar Ratio on pH Values (Perfect values)')
    plt.xlabel('Molar Ratio')
    plt.ylabel('pH Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    ## Experimentally Determined Titration Curve

    ![Experimentally determined titration curve for citric acid](https://www.researchgate.net/profile/Stephan-Schwoebel/publication/351108949/figure/fig7/AS:1017157349564433@1619520618739/Experimentally-determined-titration-curve-for-citric-acid.ppm)

    [Source](https://www.researchgate.net/figure/Experimentally-determined-titration-curve-for-citric-acid_fig7_351108949)

    Fairly similar
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, phs_ratio_fig):
    mo.md(
        rf"""
    ## Effect of Experimental pHs on specitaion fractions

    {mo.as_html(phs_ratio_fig)}
    """
    )
    return


@app.cell
def _(corrected_pka, graph_molarity, phfork, phs, plt):
    citricacid = phfork.AcidAq(
        pKa=corrected_pka, charge=0, conc=graph_molarity
    )

    fracs = citricacid.alpha(phs)
    print(fracs[0])

    fig = plt.figure()

    plt.plot(phs, fracs)
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    phs_ratio_fig = plt.gca()

    return fracs, phs_ratio_fig


@app.cell
def _(chemicalshift_fig, mo):
    mo.md(
        rf"""
    ## Speciation Ratios Against Measured Chemical Shifts

    {mo.as_html(chemicalshift_fig)}
    """
    )
    return


@app.cell
def _(np):
    """
    Obtain chemical shifts and SR values from source
    """

    import re
    import xml.etree.ElementTree as ET
    import os

    def type_check(**type_hints):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for arg_name, expected_type in type_hints.items():
                    if arg_name in kwargs:
                        arg_value = kwargs[arg_name]
                    else:
                        arg_index = list(type_hints.keys()).index(arg_name)
                        arg_value = args[arg_index]

                    if isinstance(expected_type, type) and not isinstance(arg_value, expected_type):
                        raise TypeError(f'Expected {arg_name} to be of type {expected_type.__name__}, got {type(arg_value).__name__}')

                    # Check for list of specific type
                    if isinstance(expected_type, tuple) and expected_type[0] == list:
                        if not isinstance(arg_value, list):
                            raise TypeError(f'Expected {arg_name} to be a list, got {type(arg_value).__name__}')
                        for item in arg_value:
                            if not isinstance(item, (expected_type[1], np.float32, np.float64)):
                                raise TypeError(f'All items in {arg_name} must be of type {expected_type[1].__name__}, got {type(item).__name__}')
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @type_check(base_dir=str, experiment_dir=str, count=int)
    def get_experiment_directories(base_dir, experiment_dir, count):
        """Generate a list of experiment directories."""
        return [f'{experiment_dir}_{i}' for i in range(1, count + 1)]

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def extract_sfo1_and_o1_values(experiment, experiment_number, data_dir):
        """Extract SFO1 and O1 values from the acqus file."""

        dir = f'{data_dir}/{experiment}/{experiment_number}'

        if os.path.exists(dir) is False:
            raise FileNotFoundError(f'Directory {dir} does not exist')

        if os.path.exists(f'{dir}/acqus') is False:
            raise FileNotFoundError(f'Directory {dir} does not contain acqusition paramaters')

        sfo1, o1 = None, None

        with open(f'{dir}/acqus') as f:
            for line in f:
                match1 = re.search(r'\$SFO1=\s*([\d.]+)', line)
                match2 = re.search(r'\$O1=\s+([\d.]+)', line)
                if match1:
                    sfo1 = np.float64(match1.group(1))
                if match2:
                    o1 = np.float64(match2.group(1))

        if sfo1 is None:
            raise ValueError(f'No sfo1 value in directory {dir}')

        if o1 is None:
            raise ValueError(f'No o1 value in directory {dir}')

        return sfo1, o1

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def extract_sf_values(experiment, experiment_number, data_dir):
        """Extract SF values from the procs file."""

        dir = f'{data_dir}/{experiment}/{experiment_number}'

        if os.path.exists(dir) is False:
            raise FileNotFoundError(f'Directory {dir} does not exist')

        if os.path.exists(f'{dir}/pdata/1/procs') is False:
            raise FileNotFoundError(f'Directory {dir} does not contain procs paramaters')

        sf = None

        with open(f'{dir}/pdata/1/procs') as f:
            for line in f:
                match = re.search(r'\$SF=\s*([\d.]+)', line)
                if match:
                    sf = np.float64(match.group(1))

        if sf is None:
            raise ValueError(f'No sr value in directory {dir}')

        return sf

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def extract_peak_values(experiment, experiment_number, data_dir):
        """Extract peak values from the peaklist.xml file."""
        peak_values = []

        dir = f'{data_dir}/{experiment}/{experiment_number}'

        if os.path.exists(dir) is False:
            raise FileNotFoundError(f'Directory {dir} does not exist')

        if os.path.exists(f'{dir}/pdata/1/peaklist.xml') is False:
            raise FileNotFoundError(f'Directory {dir} does not contain a peaklist')

        with open(f'{data_dir}/{experiment}/{experiment_number}/pdata/1/peaklist.xml') as f:
            xml_data = f.read()
            root = ET.fromstring(xml_data)

            for peak in root.findall('.//Peak1D'):
                f1 = np.float64(peak.get('F1'))
                intensity = np.float64(peak.get('intensity'))
                peak_values.append([f1, intensity])

        if len(peak_values) is 0:
            raise ValueError(f'No peaklist values in directory {dir}.')

        return peak_values

    @type_check(sr=float, frequency=float)
    def calculate_ppm_shift(sr, frequency):
        """Calculate PPM shift from SR."""
        return sr / frequency

    @type_check(peak_values=(list, list, float), ppm_shit =(list, float))
    def adjust_peak_values(peak_values, ppm_shift):
        """Adjust peak values based on PPM shift."""
        for idx, peaks in enumerate(peak_values):
            for peak in peaks:
                peak[0] += ppm_shift[idx]

    @type_check(o1=float, sf=float, sfo1=float)
    def calculate_sr(o1, sf, sfo1):
        return o1 + sf * 1e6 - sfo1 * 1e6

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def extract_sr(experiment, experiment_number, data_dir):
        sfo1, o1 = extract_sfo1_and_o1_values(experiment, experiment_number, data_dir)
        sf = extract_sf_values(experiment, experiment_number, data_dir)
        sr = calculate_sr(o1, sf, sfo1)
        return sr

    data_dir = 'spectra' # The directory all data is in
    experiment_dir = '20250811_cit_nacit_titr' # The experiment name
    experiment_count = 24 # The number of experiments in format _i
    experiment_number = '3' # The folder in the experiment that contains the acqusition data

    experiments = get_experiment_directories(data_dir, experiment_dir, experiment_count)

    sr_values, peak_values = [], []

    for experiment in experiments:
        sr_values.append(extract_sr(experiment, experiment_number, data_dir))

        peaks = extract_peak_values(experiment, experiment_number, data_dir)
        peak_values.append(peaks)

    ppm_shift = []

    for sr in sr_values:
        ppm_shift.append(calculate_ppm_shift(sr, frequency=600.5))

    adjust_peak_values(peak_values, ppm_shift)

    # # Uncomment to print results
    # print(sr_values)
    # print(ppm_shift)
    # print(peak_values[0])
    return data_dir, experiment_number, experiments, peak_values, type_check


@app.cell
def _(fracs):
    species_1 = []
    species_2 = []
    species_3 = []
    species_4 = []

    for i in fracs:
        species_1.append(i[0])
        species_2.append(i[1])
        species_3.append(i[2])
        species_4.append(i[3])

    print(species_1[0] * 100)
    return species_1, species_2, species_3, species_4


@app.cell
def _(peak_values, plt, species_1, species_2, species_3, species_4):
    def _1():
        avg_ppm = []
        for peaks in peak_values:
            average = 0
            for peak in peaks:
                average += peak[0]

            average /= len(peaks)

            avg_ppm.append(average)
        return avg_ppm

    avg_ppm = _1()



    plt.figure(figsize=(14, 8))

    plt.plot(avg_ppm, species_1)
    plt.plot(avg_ppm, species_2)
    plt.plot(avg_ppm, species_3)
    plt.plot(avg_ppm, species_4)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.ylabel('Speciation Ratio')
    plt.xlabel('Chemical shift (PPM)')
    plt.title('Chemical shift of Peaks in Citric Acid and Trisodium Citrate Speciation')
    chemicalshift_fig = plt.gca()
    return (chemicalshift_fig,)


@app.cell
def _(fidfig, mo, singlefidfig):
    mo.md(
        rf"""
    ## FID Spectra from Experimental
    ### All

    {mo.as_html(fidfig)}

    ### Single, higher resolution


    {mo.as_html(singlefidfig)}
    """
    )
    return


@app.cell
def _(data_dir, experiment_number, experiments, np, plt, type_check):
    import struct
    import math
    import seaborn as sns

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def read_fid(experiment, experiment_number, data_dir):
        dir = f'{data_dir}/{experiment}/{experiment_number}/fid'
        with open(dir, 'rb') as fid_file:
            # Read the first few bytes to determine the data type
            # This is a placeholder; you need to implement the logic to read DTYPA and NC
            dtypa = "int"  # or "double", based on your file
            nc = 0  # Set this based on your file's parameters

            # Read the entire file into a byte array
            fid_data = fid_file.read()

            if dtypa == "int":
                # Calculate the number of data points
                num_points = len(fid_data) // 4  # 4 bytes for each int
                data = np.zeros(num_points, dtype=np.int32)

                for i in range(num_points):
                    data[i] = struct.unpack('i', fid_data[i*4:(i+1)*4])[0]

                # Apply the exponent
                data = data * (10 ** nc)

            elif dtypa == "double":
                num_points = len(fid_data) // 8  # 8 bytes for each double
                data = np.zeros(num_points, dtype=np.float64)

                for i in range(num_points):
                    data[i] = struct.unpack('d', fid_data[i*8:(i+1)*8])[0]

            return data


    def plot_fid_experiments(experiments, experiment_number, data_dir):
        # Set the style for the plots
        sns.set(style="whitegrid")  # Use Seaborn's whitegrid style for a clean look

        # Create a new figure
        plt.figure(figsize=(12, 10))

        for idx, experiment in enumerate(experiments):
            # Read the FID data
            data = read_fid(experiment, experiment_number, data_dir)

            # Calculate the number of rows and columns for subplots
            n = len(experiments)
            rows = round(math.sqrt(n))
            cols = round(math.ceil(n / rows))

            plt.subplot(rows, cols, idx + 1)
            # plt.plot(data, marker='o', linestyle='-', color=sns.color_palette("husl", n_colors=n)[idx], linewidth=2, markersize=5)
            plt.plot(data, linestyle='-', color=sns.color_palette("husl", n_colors=n)[idx], linewidth=0.5, markersize=5)

            # Add titles and labels
            plt.title(f'FID Experiment {idx + 1}', fontsize=14)
            plt.xlabel('Time (ms)', fontsize=12)  # Replace with actual time unit if different
            plt.ylabel('Magnitude', fontsize=12)  # Magnitude of FID data
            plt.grid(True)  # Add grid lines for better readability
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.suptitle('FID Experiments Overview', fontsize=16, y=1.02)  # Main title for the figure
        plt.savefig('figs/FID.svg')
        return plt.gca()  # Return the current axes

    # Example usage
    fidfig = plot_fid_experiments(experiments, experiment_number, data_dir)

    data = read_fid(experiments[0], experiment_number, data_dir)

    return data, fidfig


@app.cell
def _(data, plt):
    plt.plot(data, linestyle='-', linewidth=0.5, markersize=5)
    plt.savefig('figs/singleFID.svg')
    singlefidfig = plt.gca()
    return (singlefidfig,)


if __name__ == "__main__":
    app.run()
