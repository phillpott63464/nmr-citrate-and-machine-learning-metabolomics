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


@app.cell(hide_code=True)
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
    """Load pHs and correct for ionic strengths"""

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

    def calculate_ionic_strength_from_ratios(total_conc, ratios):
        ionic_strength = 0
        for ratio in ratios:
            # Calculate concentrations based on the total concentration and ratios
            acid_ratio = ratio['acid ratio'] / 100 * total_conc
            base_ratio = ratio['base ratio'] / 100 * total_conc

            # Concentrations of each species
            H3A_conc = acid_ratio * (1 - base_ratio / total_conc)  # H3A
            H2A_conc = acid_ratio * (base_ratio / total_conc)      # H2A-
            HA2_conc = base_ratio / total_conc                      # HA2-
            A3_conc = base_ratio / total_conc                       # A3-

            # Charges of each species
            charges = [0, -1, -2, -3]  # H3A, H2A-, HA2-, A3-
            concentrations = [H3A_conc, H2A_conc, HA2_conc, A3_conc]

            # Calculate ionic strength contribution
            for c, z in zip(concentrations, charges):
                ionic_strength += c * (z**2)

        return 0.5 * ionic_strength

    def debeye_huckel_log_gamma(z, I):
        """Davies equation log10 gamma"""
        sqrt_I = np.sqrt(I)
        return -A_CONST * z**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)

    def correct_pkas(pkas, old_molarity, new_molarity, charges):
        """Adjust each pKa from old ionic strength to new ionic strength."""
        # I_old = ionic_strength_from_conc(old_molarity)
        # I_new = ionic_strength_from_conc(new_molarity)

        old_ratios = simulate_ph_graph(pkas, old_molarity, charge=0)
        new_ratios = simulate_ph_graph(pkas, new_molarity, charge=0)

        I_old = calculate_ionic_strength_from_ratios(old_molarity, old_ratios)
        I_new = calculate_ionic_strength_from_ratios(new_molarity, new_ratios)

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
    return (
        corrected_pka,
        graph_molarity,
        np,
        phfork,
        pkasolver,
        simulate_ph_graph,
    )


@app.cell
def _(corrected_pka, mo, np, pkasolver, simulate_ph_graph):
    import pandas as pd

    out_dir = 'experimental'
    acid_molecular_weight = 192.12   # g/mol
    base_molecular_weight = 258.07   # g/mol
    dss_molecular_weight = 224.36   # g/mol

    imported = pd.read_csv(f'{out_dir}/na-eppendorfs.csv')
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

    ratios = simulate_ph_graph(
        corrected_pka,
        ((stocks['acid']['molarity'] + stocks['base']['molarity']) / 2),
    )

    pkasolver_ratios = simulate_ph_graph(
        pkasolver,
        ((stocks['acid']['molarity'] + stocks['base']['molarity']) / 2),
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

    a = calculate_ratio(5)   # 5% D2O
    phAlter = 0.3139 * a + 0.0854 * a**2

    for idx, x in enumerate(expected_phs):
        expected_phs[idx] += phAlter

    for idx, x in enumerate(pkasolver_phs):
        pkasolver_phs[idx] += phAlter

    output = mo.ui.table(
        data=[
            {
                'Experiment': f'Experiment {idx + 1}',
                'Total Volume (ml)': round(x * 1000, 2),
                'pH': y,
                'Expected pH': round(d, 2),
                'Acid Ratio': round(((z * 1000) / 0.6), 2),
                'Base Ratio': round(((i * 1000) / 0.6), 2),
            }
            for idx, (x, y, z, i, d) in enumerate(
                zip(total_vol, phs, acid_vol, base_vol, expected_phs)
            )
        ],
        label='Experiment Data',
    )

    return (
        acid_vol,
        base_vol,
        expected_acid_ratios,
        expected_phs,
        out_dir,
        output,
        pd,
        phs,
        pkasolver_phs,
        pkasolver_ratios,
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
    expected_moles_acid = [
        ratio * 0.0006 * stocks['acid']['molarity']
        for ratio in expected_acid_ratios
    ]
    expected_moles_base = [
        (1 - ratio) * 0.0006 * stocks['acid']['molarity']
        for ratio in expected_acid_ratios
    ]

    # Calculate molar ratios
    # molar_ratios = [acid / base if base != 0 else 7 for acid, base in zip(moles_acid, moles_base)]
    molar_ratios = [
        (base - acid) for acid, base in zip(moles_acid, moles_base)
    ]

    expected_molar_ratios = [
        (base - acid)
        for acid, base in zip(expected_moles_acid, expected_moles_base)
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(molar_ratios, phs, label='Experimental pHs', marker='o')
    plt.plot(
        expected_molar_ratios, expected_phs, label='Expected pHs', marker='x'
    )
    plt.plot(
        expected_molar_ratios, pkasolver_phs, label='Pkasolver pHs', marker='x'
    )

    for id, point in enumerate(corrected_pka):
        plt.axhline(
            y=point, color='red', linestyle='--', label=f'pka{id+1} = {point}'
        )

    plt.title('Effect of Molar Ratio on pH Values')
    plt.xlabel('Molar Ratio')
    plt.ylabel('pH Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

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
    plt.plot(
        molar_ratios_perfect,
        [x['pH'] for x in ratios_perfect],
        label='Solved pHs',
    )
    # plt.plot(moles_acid_perfect, [x['pH'] for x in ratios_perfect])

    plt.plot(
        molar_ratios_perfect,
        [x['pH'] for x in pkasolver_ratios],
        label='Pkasolver pHs',
    )

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

                    if isinstance(expected_type, type) and not isinstance(
                        arg_value, expected_type
                    ):
                        raise TypeError(
                            f'Expected {arg_name} to be of type {expected_type.__name__}, got {type(arg_value).__name__}'
                        )

                    # Check for list of specific type
                    if (
                        isinstance(expected_type, tuple)
                        and expected_type[0] == list
                    ):
                        if not isinstance(arg_value, list):
                            raise TypeError(
                                f'Expected {arg_name} to be a list, got {type(arg_value).__name__}'
                            )
                        for item in arg_value:
                            if not isinstance(
                                item,
                                (expected_type[1], np.float32, np.float64),
                            ):
                                raise TypeError(
                                    f'All items in {arg_name} must be of type {expected_type[1].__name__}, got {type(item).__name__}'
                                )
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
            raise FileNotFoundError(
                f'Directory {dir} does not contain acqusition paramaters'
            )

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
            raise FileNotFoundError(
                f'Directory {dir} does not contain procs paramaters'
            )

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
            raise FileNotFoundError(
                f'Directory {dir} does not contain a peaklist'
            )

        with open(
            f'{data_dir}/{experiment}/{experiment_number}/pdata/1/peaklist.xml'
        ) as f:
            xml_data = f.read()
            root = ET.fromstring(xml_data)

            for peak in root.findall('.//Peak1D'):
                f1 = np.float64(peak.get('F1'))
                intensity = np.float64(peak.get('intensity'))
                peak_values.append([f1, intensity])

        if len(peak_values) == 0:
            raise ValueError(f'No peaklist values in directory {dir}.')

        return peak_values

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def extract_phc(experiment, experiment_number, data_dir):
        """Extract Phase Correction values from the procs file."""

        dir_path = f'{data_dir}/{experiment}/{experiment_number}'

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f'Directory {dir_path} does not exist')

        procs_file_path = f'{dir_path}/pdata/1/procs'
        if not os.path.exists(procs_file_path):
            raise FileNotFoundError(
                f'Directory {dir_path} does not contain procs parameters'
            )

        phc = {}

        with open(procs_file_path) as f:
            for line in f:
                match = re.search(r'\$PHC(\d*)=\s*([-+]?\d*\.\d+|\d+)', line)
                if match:
                    key = f'PHC{match.group(1)}' if match.group(1) else 'PHC'
                    phc[key] = np.float64(match.group(2))

        if not phc:
            raise ValueError(
                f'No phase correction values in directory {dir_path}'
            )

        return phc

    @type_check(sr=float, frequency=float)
    def calculate_ppm_shift(sr, frequency):
        """Calculate PPM shift from SR."""
        return sr / frequency

    @type_check(peak_values=(list, list, float), ppm_shit=(list, float))
    def adjust_peak_values(peak_values, ppm_shift):
        """Adjust peak values based on PPM shift."""
        for idx, peaks in enumerate(peak_values):
            for peak in peaks:
                peak[0] += ppm_shift[idx]

        return peak_values

    @type_check(o1=float, sf=float, sfo1=float)
    def calculate_sr(o1, sf, sfo1):
        return o1 + sf * 1e6 - sfo1 * 1e6

    @type_check(experiment=str, experiment_number=str, data_dir=str)
    def extract_sr(experiment, experiment_number, data_dir):
        sfo1, o1 = extract_sfo1_and_o1_values(
            experiment, experiment_number, data_dir
        )
        sf = extract_sf_values(experiment, experiment_number, data_dir)
        sr = calculate_sr(o1, sf, sfo1)
        return sr

    data_dir = 'spectra'   # The directory all data is in
    experiment_dir = '20250811_cit_nacit_titr'   # The experiment name
    experiment_count = 24   # The number of experiments in format _i
    experiment_number = (
        '3'  # The folder in the experiment that contains the acqusition data
    )

    experiments = get_experiment_directories(
        data_dir, experiment_dir, experiment_count
    )

    sr_values, peak_values = [], []

    for experiment in experiments:
        sr_values.append(extract_sr(experiment, experiment_number, data_dir))

        peaks = extract_peak_values(experiment, experiment_number, data_dir)
        peak_values.append(peaks)

    ppm_shift = []

    for sr in sr_values:
        ppm_shift.append(calculate_ppm_shift(sr, frequency=600.5))

    peak_values = adjust_peak_values(peak_values, ppm_shift)

    # # Uncomment to print results
    # print(sr_values)
    # print(ppm_shift)
    # print(peak_values[0])
    return (
        data_dir,
        experiment_number,
        experiments,
        extract_peak_values,
        extract_phc,
        peak_values,
    )


@app.cell(hide_code=True)
def _(chemicalshift_ph_fig, mo):
    mo.md(
        rf"""
    ## Graphs:

    ### Chemical Shift Against pH/base ratio

    {mo.as_html(chemicalshift_ph_fig)}
    """
    )
    return


@app.cell
def _(base_vol, peak_values, phs, plt):
    def _1():
        avg_ppm = []
        for peaks in peak_values:
            average = 0
            for peak in peaks:
                # print(peak)
                average += peak[0]

            average /= len(peaks)

            avg_ppm.append(average)
        return avg_ppm

    avg_ppm = _1()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot([x / 0.0006 * 100 for x in base_vol], avg_ppm)

    plt.ylabel('Average PPM')
    plt.xlabel('Sodium Citrate Percentage')
    plt.title('Sodium Citrate Percentage and Average PPM')

    plt.subplot(1, 3, 2)
    plt.plot(phs, avg_ppm)

    plt.ylabel('Average PPM')
    plt.xlabel('pH')
    plt.title('pH and Average PPM')

    def _2():
        avg_ppms = []
        for peaks in peak_values:
            avg_ppms.append(
                [
                    (peaks[0][0] + peaks[1][0]) / 2,
                    (peaks[2][0] + peaks[3][0]) / 2,
                ]
            )

        return avg_ppms

    avg_ppms = _2()

    plt.subplot(1, 3, 3)
    plt.plot(phs, avg_ppms)

    plt.ylabel('Average PPM Per Peak')
    plt.xlabel('pH')
    plt.title('pH and Average PPM Per Peak')
    plt.legend(
        [
            'Peak 2 (furthest from reference at 0)',
            'Peak 1 (closest to reference at 0)',
        ]
    )

    plt.tight_layout()

    chemicalshift_ph_fig = plt.gca()
    return avg_ppm, chemicalshift_ph_fig


@app.cell(hide_code=True)
def _(chemicalshift_fig, mo):
    mo.md(
        rf"""
    ### Chemical Shift Against Citrate Speciation

    {mo.as_html(chemicalshift_fig)}
    """
    )
    return


@app.cell
def _(avg_ppm, base_vol, corrected_pka, graph_molarity, phfork, phs, plt):
    citricacid = phfork.AcidAq(
        pKa=corrected_pka, charge=0, conc=graph_molarity
    )

    fracs = citricacid.alpha(phs)

    species_1 = []
    species_2 = []
    species_3 = []
    species_4 = []

    for i in fracs:
        species_1.append(i[0])
        species_2.append(i[1])
        species_3.append(i[2])
        species_4.append(i[3])

    print([float(x) for x in species_1])
    print([float(x) for x in species_2])
    print([float(x) for x in species_3])
    print([float(x) for x in species_4])

    print(species_1[0] * 100)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(
        [x / 0.0006 * 100 for x in base_vol],
        fracs,
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.ylabel('Speciation Ratio')
    plt.xlabel('Sodium Citrate Percentage')
    plt.title('Sodium Citrate Percentage and Trisodium Citrate Speciation')

    plt.subplot(1, 3, 2)
    plt.plot(
        phs,
        fracs,
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.ylabel('Speciation Ratio')
    plt.xlabel('pH')
    plt.title('pH and Trisodium Citrate Speciation')

    plt.subplot(1, 3, 3)
    plt.plot(
        avg_ppm,
        fracs,
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.gca().invert_xaxis()
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.ylabel('Speciation Ratio')
    plt.xlabel('Chemical shift (PPM)')
    plt.title(
        'Chemical shift of Peaks in Citric Acid and Trisodium Citrate Speciation'
    )

    plt.tight_layout()

    chemicalshift_fig = plt.gca()

    print([x / 0.0006 * 100 for x in base_vol])
    return chemicalshift_fig, fracs


@app.cell(hide_code=True)
def _(citratecouplingfig, mo):
    mo.md(
        rf"""
    ### Citrate J Coupling Against Citrate Speciation

    The value between the peaks for each proton

    {mo.as_html(citratecouplingfig)}
    """
    )
    return


@app.cell
def _(base_vol, citrate_ppms, fracs, np, phs, plt):
    citrate_couplings_raw = [
        [
            x[0] - x[1],
            x[2] - x[3],
        ]
        for x in citrate_ppms
    ]   # Calculate j coupling values

    citrate_couplings = [
        np.average(x) for x in citrate_couplings_raw
    ]   # Average to a single value

    plt.figure(figsize=(15, 5))

    # First subplot
    plt.subplot(1, 3, 1)
    plt.plot(
        [x / 0.0006 * 100 for x in base_vol],
        citrate_couplings,
        color='blue',
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.title('J Coupling vs. Sodium Citrate Percentage', fontsize=14)
    plt.xlabel('Sodium Citrate Percentage', fontsize=12)
    plt.ylabel('J Coupling', fontsize=12)
    plt.grid(True)

    # Second subplot
    plt.subplot(1, 3, 2)
    plt.plot(
        phs,
        citrate_couplings,
        color='green',
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.title('J Coupling vs. pH', fontsize=14)
    plt.xlabel('pH', fontsize=12)
    plt.ylabel('J Coupling', fontsize=12)
    plt.grid(True)

    # Third subplot
    plt.subplot(1, 3, 3)
    plt.plot(
        citrate_couplings,
        fracs,
        marker='o',
        linestyle='-',
        linewidth=2,
    )
    plt.title('J Coupling vs. Citrate Species', fontsize=14)
    plt.xlabel('J Coupling', fontsize=12)
    plt.ylabel('Speciation Ratio', fontsize=12)
    plt.gca().invert_xaxis()
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    print(phs)
    print([float(x) for x in citrate_couplings])

    # Show the plots
    citratecouplingfig = plt.gca()
    return (citratecouplingfig,)


@app.cell(hide_code=True)
def _(citratepeakdifferencesfig, mo):
    mo.md(
        rf"""
    ### Citrate Peak Differences Against Citrate Speciation

    The value between the two protons

    {mo.as_html(citratepeakdifferencesfig)}
    """
    )
    return


@app.cell
def _(
    base_vol,
    data_dir,
    experiment_number,
    experiments,
    extract_peak_values,
    fracs,
    np,
    phs,
    plt,
):
    citrate_peaks = [
        extract_peak_values(
            data_dir=data_dir,
            experiment_number=experiment_number,
            experiment=experiment,
        )
        for experiment in experiments
    ]

    # print(len(citrate_ppms[0]))

    citrate_ppms = [
        [x[0] for x in y] for y in citrate_peaks
    ]   # Discard intensities, not required for this

    # print(len(citrate_ppms[0]))

    citrate_shifts = [
        [
            np.average(x[0:1]),
            np.average(x[2:3]),
        ]
        for x in citrate_ppms
    ]   # Average the multiplets together

    # print(len(citrate_shifts[0]))

    citrate_differences = [
        float(round(x[0] - x[1], 4)) for x in citrate_shifts
    ]

    # print(f'{citrate_differences}')

    # Create a figure with a specific size
    plt.figure(figsize=(15, 5))

    # First subplot
    plt.subplot(1, 3, 1)
    plt.plot(
        [x / 0.0006 * 100 for x in base_vol],
        citrate_differences,
        color='blue',
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.title('Peak Differences vs. Sodium Citrate Percentage', fontsize=14)
    plt.xlabel('Sodium Citrate Percentage', fontsize=12)
    plt.ylabel('Peak Differences', fontsize=12)
    plt.grid(True)

    # Second subplot
    plt.subplot(1, 3, 2)
    plt.plot(
        phs,
        citrate_differences,
        color='green',
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.title('Peak Differences vs. pH', fontsize=14)
    plt.xlabel('pH', fontsize=12)
    plt.ylabel('Peak Differences', fontsize=12)
    plt.grid(True)

    # Third subplot
    plt.subplot(1, 3, 3)
    plt.plot(
        citrate_differences,
        fracs,
        marker='o',
        linestyle='-',
        linewidth=2,
    )

    plt.title('Peak Differences vs. Citrate Species', fontsize=14)
    plt.xlabel('Peak Differences', fontsize=12)
    plt.ylabel('Speciation Ratio', fontsize=12)
    plt.gca().invert_xaxis()
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    citratepeakdifferencesfig = plt.gca()

    return citrate_ppms, citratepeakdifferencesfig


@app.cell(hide_code=True)
def _(fidfig, mo):
    mo.md(
        rf"""
    ## FID Spectra from Experimental
    ### All

    {mo.as_html(fidfig)}
    """
    )
    return


@app.cell
def _(data_dir, experiment_number, experiments, math, plt, read_bruker):
    import struct
    import seaborn as sns

    def plot_fid_experiments(experiments, experiment_number, data_dir):
        # Set the style for the plots
        sns.set(
            style='whitegrid'
        )  # Use Seaborn's whitegrid style for a clean look

        # Create a new figure
        plt.figure(figsize=(12, 10))

        for idx, experiment in enumerate(experiments):
            # Read the FID data
            data = read_bruker(data_dir, experiment, experiment_number)

            # Calculate the number of rows and columns for subplots
            n = len(experiments)
            rows = round(math.sqrt(n))
            cols = round(math.ceil(n / rows))

            plt.subplot(rows, cols, idx + 1)
            # plt.plot(data, marker='o', linestyle='-', color=sns.color_palette("husl", n_colors=n)[idx], linewidth=2, markersize=5)
            plt.plot(
                data,
                linestyle='-',
                color=sns.color_palette('husl', n_colors=n)[idx],
                linewidth=0.5,
                markersize=5,
            )

            # Add titles and labels
            plt.title(f'FID Experiment {idx + 1}', fontsize=14)
            plt.xlabel(
                'Time (ms)', fontsize=12
            )  # Replace with actual time unit if different
            plt.ylabel('Magnitude', fontsize=12)  # Magnitude of FID data
            plt.grid(True)  # Add grid lines for better readability
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.suptitle(
            'FID Experiments Overview', fontsize=16, y=1.02
        )  # Main title for the figure
        plt.savefig('figs/FID.svg')
        return plt.gca()  # Return the current axes

    # Example usage
    fidfig = plot_fid_experiments(experiments, experiment_number, data_dir)
    return (fidfig,)


@app.cell(hide_code=True)
def _(mo, singlefidfig):
    mo.md(
        rf"""
    ### Single, higher resolution

    {mo.as_html(singlefidfig)}
    """
    )
    return


@app.cell
def _(data_dir, experiment_number, experiments, plt, read_bruker):
    fiddata = read_bruker(
        data_dir=data_dir,
        experiment=experiments[0],
        experiment_number=experiment_number,
    )

    plt.plot(fiddata, linestyle='-', linewidth=0.5, markersize=5)
    plt.savefig('figs/singleFID.svg')
    singlefidfig = plt.gca()
    return (singlefidfig,)


@app.cell(hide_code=True)
def _(mo, nmrgluefig):
    mo.md(rf"""{mo.as_html(nmrgluefig)}""")
    return


@app.cell
def _(data_dir, experiment_number, experiments, extract_phc, plt):
    def log(msg):
        with open('log.log', 'a') as f:
            if isinstance(msg, list):
                for x in msg:
                    log(x)  # Recursively log each item in the list
            else:
                f.writelines(
                    str(msg) + '\n'
                )  # Convert msg to string and add a newline

    def bruker_fft(data_dir, experiment, experiment_number):
        """Convert time domain data to frequency domain"""
        import nmrglue as ng

        phc = extract_phc(
            data_dir=data_dir,
            experiment_number=experiment_number,
            experiment=experiment,
        )

        data = read_bruker(data_dir, experiment, experiment_number)
        # data = read_fid(data_dir=data_dir, experiment=experiment, experiment_number=experiment_number)

        # Process the spectrum
        data = ng.proc_base.zf_size(
            data, 2**15
        )    # Zero fill to 32768 points
        data = ng.proc_base.fft(data)                 # Fourier transform
        data = ng.proc_base.ps(
            data, p0=phc['PHC0'], p1=phc['PHC1']
        )  # Phase correction
        data = ng.proc_base.di(data)                  # Discard the imaginaries
        data = ng.proc_base.rev(data)                 # Reverse the data=

        return data

    def read_bruker(data_dir, experiment, experiment_number):
        import nmrglue as ng

        phc = extract_phc(
            data_dir=data_dir,
            experiment_number=experiment_number,
            experiment=experiment,
        )

        dic, data = ng.bruker.read(
            f'{data_dir}/{experiment}/{experiment_number}'
        )
        # Remove the digital filter
        data = ng.bruker.remove_digital_filter(dic, data)

        return data

    import math

    def _():
        # Create a new figure
        ngfig = plt.figure(figsize=(12, 10))

        # Calculate the number of rows and columns for subplots
        n = len(experiments)
        rows = round(math.sqrt(n))
        cols = round(math.ceil(n / rows))

        for idx, experiment in enumerate(experiments):
            data = bruker_fft(
                data_dir=data_dir,
                experiment=experiment,
                experiment_number=experiment_number,
            )

            ax = ngfig.add_subplot(rows, cols, idx + 1)
            ax.plot(data[19000:22000])  # Adjust the range as needed
            # ax.plot(data)
            ax.set_title(f'NMR Experiment {idx + 1}', fontsize=14)
            ax.set_xlabel(
                'Data Points', fontsize=12
            )  # Replace with actual x-axis label if needed
            ax.set_ylabel('Magnitude', fontsize=12)     # Magnitude of NMR data
            ax.grid(True)  # Add grid lines for better readability
            ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.suptitle(
            'NMR Experiments Overview', fontsize=16, y=1.02
        )  # Main title for the figure
        plt.savefig('figs/NMR.svg')
        return plt.gca()  # Return the current axes

    nmrgluefig = _()
    return bruker_fft, math, nmrgluefig, read_bruker


@app.cell
def _(mo, singlefftfig):
    mo.md(
        rf"""
    ## Single FFT, higher resolution

    {mo.as_html(singlefftfig)}
    """
    )
    return


@app.cell
def _(bruker_fft, data_dir, experiment_number, experiments, plt):
    def _():
        # plt.figure(figsize=(15, 5))

        data = bruker_fft(
            data_dir=data_dir,
            experiment=experiments[8],
            experiment_number=experiment_number,
        )

        plt.plot(data[19000:22000])
        plt.title(f'NMR Experiment', fontsize=14)
        plt.xlabel(
            'Data Points', fontsize=12
        )  # Replace with actual x-axis label if needed
        plt.ylabel('Magnitude', fontsize=12)     # Magnitude of NMR data
        plt.grid(True)  # Add grid lines for better readability

        plt.legend()

        plt.tight_layout()  # Adjust layout to prevent overlap

        return plt.gca()

    singlefftfig = _()

    # singlefftfig.show()
    return (singlefftfig,)


@app.cell
def _():
    return


@app.cell
def _():
    # 4 protons, because it's a dicitrate
    uranium_proton_ppms = [4.08, 4.19, 4.26, 4.35]

    uranium_proton_ppms_converted = [
        x * 220 / 600.05 for x in uranium_proton_ppms
    ]

    print(
        f'Uranium chemical shifts: {[round(x, 2) for x in uranium_proton_ppms_converted]}'
    )
    print(
        f'Uranium difference between peaks: {[abs(round(x, 2)) for x in [uranium_proton_ppms_converted[0] - uranium_proton_ppms_converted[1], uranium_proton_ppms_converted[2] - uranium_proton_ppms_converted[3]]]}'
    )

    return


@app.cell
def _():
    """Metal Experiments load"""

    magnesium_chloride_mass = 95.21   # g/mol
    calcium_chloride_mass = 110.98   # g/mol
    tris_mass = 121.14   # g/mol
    tris_chloride_mass = 157.59   # g/mol
    d2o_density = 1.1044   # g/ml

    tris_buffer_stock = {
        'boat tris': 0.49,
        'flask': 35.56,
        'flask and tris': 36.06,
        'boat tris hcl': 0.15,
        'flask and tris hcl': 36.19,
        'flask and tris hcl rinse': 38.87,
        'flask and d2o': 39.15,
        'flask and mq': 85.62,
    }

    tris_buffer_stock['tris'] = (
        tris_buffer_stock['flask and tris'] - tris_buffer_stock['flask']
    )

    tris_buffer_stock['tris hcl'] = (
        tris_buffer_stock['flask and tris hcl']
        - tris_buffer_stock['flask and tris']
    )

    tris_buffer_stock['d2o'] = (
        tris_buffer_stock['flask and d2o']
        - tris_buffer_stock['flask and tris hcl rinse']
    )

    tris_buffer_stock['mq'] = (
        tris_buffer_stock['flask and mq'] - tris_buffer_stock['flask and d2o']
    ) + (
        tris_buffer_stock['flask and tris hcl rinse']
        - tris_buffer_stock['flask and tris hcl']
    )

    tris_buffer_stock['tris molarity'] = (
        (tris_buffer_stock['tris'] + tris_buffer_stock['tris hcl']) / tris_mass
    ) / (tris_buffer_stock['mq'] + (tris_buffer_stock['d2o'] / d2o_density))

    tris_buffer_stock['chloride molarity'] = (
        tris_buffer_stock['tris hcl'] / tris_chloride_mass
    ) / (tris_buffer_stock['mq'] + (tris_buffer_stock['d2o'] / d2o_density))

    magnesium_chloride_stock = {
        'flask': 12.79513,
        'boat chloride': 23.61 / 1000,
        'flask and chloride': 12.81384,
        'flask and d2o': 13.08,
        'flask and mq': 17.80,
    }

    magnesium_chloride_stock['chloride'] = (
        magnesium_chloride_stock['flask and chloride']
        - magnesium_chloride_stock['flask']
    )
    magnesium_chloride_stock['d2o'] = (
        magnesium_chloride_stock['flask and d2o']
        - magnesium_chloride_stock['flask and chloride']
    )
    magnesium_chloride_stock['mq'] = (
        magnesium_chloride_stock['flask and mq']
        - magnesium_chloride_stock['flask and d2o']
    )
    magnesium_chloride_stock['molarity'] = (
        magnesium_chloride_stock['chloride'] / magnesium_chloride_mass
    ) / (
        magnesium_chloride_stock['mq']
        + (magnesium_chloride_stock['d2o'] / d2o_density)
    )

    calcium_chloride_stock = {
        'flask': 12.62002,
        'boat chloride': 36.77 / 1000,
        'flask and chloride': 12.8560,
        'flask and d2o': 12.92,
        'flask and mq': 17.70,
    }
    calcium_chloride_stock['chloride'] = (
        calcium_chloride_stock['flask and chloride']
        - calcium_chloride_stock['flask']
    )
    calcium_chloride_stock['d2o'] = (
        calcium_chloride_stock['flask and d2o']
        - calcium_chloride_stock['flask and chloride']
    )
    calcium_chloride_stock['mq'] = (
        calcium_chloride_stock['flask and mq']
        - calcium_chloride_stock['flask and d2o']
    )
    calcium_chloride_stock['molarity'] = (
        calcium_chloride_stock['chloride'] / calcium_chloride_mass
    ) / (
        calcium_chloride_stock['mq']
        + (calcium_chloride_stock['d2o'] / d2o_density)
    )

    return calcium_chloride_stock, magnesium_chloride_stock, tris_buffer_stock


@app.cell
def _(
    calcium_chloride_stock,
    magnesium_chloride_stock,
    out_dir,
    pd,
    stocks,
    tris_buffer_stock,
):
    metal_imported = pd.read_csv(f'{out_dir}/metal_eppendorfs.csv')
    metal_real_experiments = []

    for midx in range(len(metal_imported)):
        row = {}
        for col in metal_imported.columns:
            row[col] = metal_imported.at[midx, col]
        metal_real_experiments.append(row)

    for mexperiment in metal_real_experiments:
        mexperiment['citric acid stock / L'] = (mexperiment['post citric acid stock weight / g'] - mexperiment['eppendorf base weight / g']) / 1000
        mexperiment['salt stock / L'] = (mexperiment['post salt stock weight / g'] - mexperiment['post citric acid stock weight / g']) / 1000
        mexperiment['tris buffer / L'] = (mexperiment['post tris buffer weight / g'] - mexperiment['post salt stock weight / g']) / 1000
        mexperiment['mq / L'] = (mexperiment['post milliq weight / g'] - mexperiment['post tris buffer weight / g']) / 1000

        mexperiment['total vol / L'] = mexperiment['citric acid stock / L'] + mexperiment['salt stock / L'] + mexperiment['tris buffer / L'] + mexperiment['mq / L']

        # print(mexperiment['total vol / L'])
    
        # c1v1 = c2v2
        # c2=c1v2/v2
        mexperiment['citric acid molarity'] = stocks['acid']['molarity'] * mexperiment['citric acid stock / L'] / mexperiment['total vol / L']
        mexperiment['tris buffer molarity'] = tris_buffer_stock['tris molarity'] * mexperiment['citric acid stock / L'] / mexperiment['total vol / L']
        if mexperiment['Unnamed: 0'] in range(25, 37):
            salt_molarity = magnesium_chloride_stock['molarity']
        else:
            salt_molarity = calcium_chloride_stock['molarity']
        mexperiment['salt stock molarity'] = salt_molarity * mexperiment['salt stock / L'] /  mexperiment['total vol / L']

        print(mexperiment['salt stock molarity']*1000)
    return


if __name__ == "__main__":
    app.run()
