import marimo

__generated_with = "0.15.2"
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
    experiment_dir_speciation = (
        '20250811_cit_nacit_titr'  # The experiment name
    )
    experiment_count = 24   # The number of experiments in format _i
    experiment_number = (
        '3'  # The folder in the experiment that contains the acqusition data
    )

    experiments = get_experiment_directories(
        data_dir, experiment_dir_speciation, experiment_count
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
        adjust_peak_values,
        calculate_ppm_shift,
        data_dir,
        experiment_count,
        experiment_number,
        experiments,
        extract_peak_values,
        extract_phc,
        extract_sr,
        get_experiment_directories,
        peak_values,
        sr_values,
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
    return chemicalshift_fig, citricacid, fracs


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Properly Fitting Chemical Shift to Speciation

    First, assume the model:

    $\delta_{obs,peak}=f_0\delta_0+f_1\delta_1+f_2\delta_2+f_3\delta_3$

    Where $\delta_{x}$ is unknown chemical shift of each peak in each species. We assume that all 4 species contribute linearly to all 4 peaks. Per experiment, 16 total unknown values, in a 4x4 matrix.
    """
    )
    return


@app.cell
def _(chemicalshift_predicted_fig, mo):
    mo.md(rf"""{mo.as_html(chemicalshift_predicted_fig)}""")
    return


@app.cell
def _(fracs, np, peak_values):
    """Properly fitting chemical shift to speciation"""

    # fracs = ratio of species in that ph [experiments, 4(0-1)]

    peak_values_no_intensities = [
        [float(x[0]) for x in y] for y in peak_values
    ]   # Chemical shift of each peak in citrate [experiments, 4(ppm)]

    transposed_peaks = [list(x) for x in zip(*peak_values_no_intensities)]

    linalgout = [
        np.linalg.lstsq(fracs, peak, rcond=None) for peak in transposed_peaks
    ]
    linalgout = [list(x) for x in zip(*linalgout)]
    # deltas, residuals, rank, s

    all_deltas = np.array(linalgout[0])
    # deltas, here, is a matrix of delta_{x} from the above model.

    def find_peaks(all_deltas, ratios):
        out = []
        for deltas in all_deltas:
            out.append(find_peak(deltas, ratios))

        return out

    def find_peak(deltas, ratios):
        temp = 0
        for delta, ratio in zip(deltas, ratios):
            temp += delta * ratio

        return temp

    from scipy.optimize import minimize

    def find_f(all_deltas, shifts):
        n_species = all_deltas.shape[1]

        def objective(f):
            return np.sum((all_deltas @ f - shifts) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda f: np.sum(f) - 1},
        ]   # Everything must sum to 1

        bounds = [
            (0, 1)
        ] * n_species   # All species ratio must be between 0 and 1

        result = minimize(
            objective,
            np.ones(n_species) / n_species,
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            raise OSError({result.message})

        return result.x

    predicted_ratios = []
    for shifts in peak_values_no_intensities:
        predicted_ratios.append(find_f(all_deltas=all_deltas, shifts=shifts))
    return all_deltas, find_peaks, peak_values_no_intensities, predicted_ratios


@app.cell
def _(peak_values_no_intensities, phs, plt):
    plt.plot(phs, peak_values_no_intensities)
    return


@app.cell
def _(avg_ppm, plt, predicted_ratios):
    plt.figure(figsize=(8, 5))

    plt.plot(
        avg_ppm,
        predicted_ratios,
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=5,
    )
    plt.gca().invert_xaxis()
    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.ylabel('Predicted Speciation Ratio')
    plt.xlabel('Chemical shift (PPM)')
    plt.title(
        'Chemical shift of Peaks in Citric Acid and Predicted Trisodium Citrate Speciation'
    )

    plt.tight_layout()

    plt.savefig('figs/chemicalshift_predicted_fig.svg')

    chemicalshift_predicted_fig = plt.gca()

    plt.show()
    return (chemicalshift_predicted_fig,)


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
            data, _ = read_bruker(data_dir, experiment, experiment_number)

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
    fiddata, _ = read_bruker(
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
        """
        Convert time domain data to frequency domain
        https://github.com/jjhelmus/nmrglue/blob/master/examples/bruker_processed_1d/bruker_processed_1d.py
        """
        import nmrglue as ng

        phc = extract_phc(
            data_dir=data_dir,
            experiment_number=experiment_number,
            experiment=experiment,
        )

        data, dic = read_bruker(data_dir, experiment, experiment_number)
        # data = read_fid(data_dir=data_dir, experiment=experiment, experiment_number=experiment_number)

        # Process the spectrum
        data = ng.proc_base.zf_size(
            data, 2**15
        )    # Zero fill to 32768 points
        data = ng.proc_base.fft(data)                 # Fourier transform
        # data = ng.proc_base.ps(
        #     data, p0=phc['PHC0'], p1=phc['PHC1']
        # )  # Phase correction
        data = ng.proc_autophase.autops(
            data, 'peak_minima', p0=phc['PHC0'], p1=phc['PHC1']
        )   # Automatic phase correction
        data = ng.proc_base.di(data)                  # Discard the imaginaries
        data = ng.proc_base.rev(data)                 # Reverse the data=

        udic = ng.bruker.guess_udic(dic, data)
        uc = ng.fileiobase.uc_from_udic(udic)
        ppm_scale = uc.ppm_scale()

        return ppm_scale, data

    def read_bruker(data_dir, experiment, experiment_number):
        import nmrglue as ng

        dic, data = ng.bruker.read(
            f'{data_dir}/{experiment}/{experiment_number}'
        )
        # Remove the digital filter
        data = ng.bruker.remove_digital_filter(dic, data)

        return data, dic

    import math

    def _():
        # Create a new figure
        ngfig = plt.figure(figsize=(12, 10))

        # Calculate the number of rows and columns for subplots
        n = len(experiments)
        rows = round(math.sqrt(n))
        cols = round(math.ceil(n / rows))

        for idx, experiment in enumerate(experiments):
            ppmscale, data = bruker_fft(
                data_dir=data_dir,
                experiment=experiment,
                experiment_number=experiment_number,
            )

            if (
                idx == 0
            ):   # The first one doesn't flip properly for some reason
                data *= -1

            if abs(max(data)) > abs(min(data)):
                data *= -1

            ax = ngfig.add_subplot(rows, cols, idx + 1)
            ax.plot(
                ppmscale[19000:22000], data[19000:22000]
            )  # Adjust the range as needed
            # ax.plot(data)
            ax.set_title(f'NMR Experiment {idx + 1}', fontsize=14)
            ax.set_xlabel(
                'Chemical Shift / PPM', fontsize=12
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
def _(bruker_fft, data_dir, experiment_number, experiments, mo, plt):
    def _():
        # plt.figure(figsize=(15, 5))

        ppmscale, data = bruker_fft(
            data_dir=data_dir,
            experiment=experiments[16],
            experiment_number=experiment_number,
        )

        # plt.plot(data[19000:22000])
        plt.plot(ppmscale, data)
        plt.title(f'NMR Experiment', fontsize=14)
        plt.xlabel(
            'Chemical Shift / PPM', fontsize=12
        )  # Replace with actual x-axis label if needed
        plt.ylabel('Magnitude', fontsize=12)     # Magnitude of NMR data
        plt.grid(True)  # Add grid lines for better readability

        plt.legend()

        plt.tight_layout()  # Adjust layout to prevent overlap

        # return plt.gca()

        return plt.gcf()

    singlefftfig = _()

    mo.mpl.interactive(singlefftfig)
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


@app.cell(hide_code=True)
def _(metal_output, mo):
    mo.md(
        rf"""
    ## Metal Experiments Load

    {mo.as_html(metal_output)}
    """
    )
    return


@app.cell
def _():
    """Metal Experiments load"""

    magnesium_chloride_mass = 203.30   # g/mol
    calcium_chloride_mass = 147.02   # g/mol
    tris_mass = 121.14   # g/mol
    tris_chloride_mass = 157.59   # g/mol
    d2o_density = 1.1044   # g/ml

    tris_buffer_stock = {
        'boat tris / g': 0.49,
        'flask / g': 35.56,
        'flask and tris / g': 36.06,
        'boat tris hcl / g': 0.15,
        'flask and tris hcl / g': 36.19,
        'flask and tris hcl rinse / g': 38.87,
        'flask and d2o / g': 39.15,
        'flask and mq / g': 85.62,
    }

    tris_buffer_stock['tris / g'] = (
        tris_buffer_stock['flask and tris / g']
        - tris_buffer_stock['flask / g']
    )

    tris_buffer_stock['tris hcl / g'] = (
        tris_buffer_stock['flask and tris hcl / g']
        - tris_buffer_stock['flask and tris / g']
    )

    tris_buffer_stock['d2o / L'] = (
        (
            tris_buffer_stock['flask and d2o / g']
            - tris_buffer_stock['flask and tris hcl rinse / g']
        )
        / d2o_density
        / 1000
    )

    tris_buffer_stock['mq / L'] = (
        (
            tris_buffer_stock['flask and mq / g']
            - tris_buffer_stock['flask and d2o / g']
        )
        + (
            tris_buffer_stock['flask and tris hcl rinse / g']
            - tris_buffer_stock['flask and tris hcl / g']
        )
    ) / 1000

    tris_buffer_stock['tris molarity'] = (
        (tris_buffer_stock['tris / g'] + tris_buffer_stock['tris hcl / g'])
        / tris_mass
    ) / (tris_buffer_stock['mq / L'] + (tris_buffer_stock['d2o / L']))

    tris_buffer_stock['chloride molarity'] = (
        tris_buffer_stock['tris hcl / g'] / tris_chloride_mass
    ) / (tris_buffer_stock['mq / L'] + (tris_buffer_stock['d2o / L']))

    print(tris_buffer_stock['d2o / L'])
    print(tris_buffer_stock['chloride molarity'])

    magnesium_chloride_stock = {
        'flask / g': 12.79513,
        'boat chloride / g': 23.61 / 1000,
        'flask and chloride / g': 12.81384,
        'flask and d2o / g': 13.08,
        'flask and mq / g': 17.80,
    }

    magnesium_chloride_stock['chloride / g'] = (
        magnesium_chloride_stock['flask and chloride / g']
        - magnesium_chloride_stock['flask / g']
    )
    magnesium_chloride_stock['d2o / L'] = (
        (
            magnesium_chloride_stock['flask and d2o / g']
            - magnesium_chloride_stock['flask and chloride / g']
        )
        / d2o_density
        / 1000
    )   # L
    magnesium_chloride_stock['mq / L'] = (
        magnesium_chloride_stock['flask and mq / g']
        - magnesium_chloride_stock['flask and d2o / g']
    ) / 1000   # L
    magnesium_chloride_stock['molarity / M'] = (
        magnesium_chloride_stock['chloride / g'] / magnesium_chloride_mass
    ) / (
        magnesium_chloride_stock['mq / L']
        + (magnesium_chloride_stock['d2o / L'])
    )

    calcium_chloride_stock = {
        'flask / g': 12.62002,
        'boat chloride / g': 36.77 / 1000,
        'flask and chloride / g': 12.6560,
        'flask and d2o / g': 12.92,
        'flask and mq / g': 17.70,
    }
    calcium_chloride_stock['chloride / g'] = (
        calcium_chloride_stock['flask and chloride / g']
        - calcium_chloride_stock['flask / g']
    )
    calcium_chloride_stock['d2o / L'] = (
        (
            calcium_chloride_stock['flask and d2o / g']
            - calcium_chloride_stock['flask and chloride / g']
        )
        / d2o_density
        / 1000
    )
    calcium_chloride_stock['mq / L'] = (
        calcium_chloride_stock['flask and mq / g']
        - calcium_chloride_stock['flask and d2o / g']
    ) / 1000
    calcium_chloride_stock['molarity / M'] = (
        calcium_chloride_stock['chloride / g'] / calcium_chloride_mass
    ) / (
        calcium_chloride_stock['mq / L'] + (calcium_chloride_stock['d2o / L'])
    )
    return calcium_chloride_stock, magnesium_chloride_stock, tris_buffer_stock


@app.cell
def _(
    calcium_chloride_stock,
    magnesium_chloride_stock,
    mo,
    out_dir,
    pd,
    stocks,
    tris_buffer_stock,
):
    from collections import OrderedDict

    metal_imported = pd.read_csv(f'{out_dir}/metal_eppendorfs.csv')
    metal_real_experiments = []

    for midx in range(len(metal_imported)):
        row = {}
        for col in metal_imported.columns:
            row[col] = metal_imported.at[midx, col]
        row = OrderedDict(row)
        metal_real_experiments.append(row)

    for mexperiment in metal_real_experiments:
        mexperiment['Sample number'] = mexperiment.pop(
            'Unnamed: 0'
        )   # Change name of unnamed: o

        mexperiment.move_to_end(
            'Sample number', last=False
        )   # Move sample number to front

        mexperiment['citric acid stock / L'] = (
            mexperiment['post citric acid stock weight / g']
            - mexperiment['eppendorf base weight / g']
        ) / 1000

        mexperiment['salt stock / L'] = (
            mexperiment['post salt stock weight / g']
            - mexperiment['post citric acid stock weight / g']
        ) / 1000

        mexperiment['tris buffer / L'] = (
            mexperiment['post tris buffer weight / g']
            - mexperiment['post salt stock weight / g']
        ) / 1000

        mexperiment['mq / L'] = (
            mexperiment['post milliq weight / g']
            - mexperiment['post tris buffer weight / g']
        ) / 1000

        mexperiment['total vol / L'] = (
            mexperiment['citric acid stock / L']
            + mexperiment['salt stock / L']
            + mexperiment['tris buffer / L']
            + mexperiment['mq / L']
        )

        # print(mexperiment['total vol / L'])

        # c1v1 = c2v2
        # c2=c1v1/v2
        mexperiment['citric acid moles'] = (
            stocks['acid']['molarity'] * mexperiment['citric acid stock / L']
        )

        mexperiment['citric acid molarity / M'] = (
            mexperiment['citric acid moles'] / mexperiment['total vol / L']
        )

        mexperiment['tris buffer molarity / M'] = (
            tris_buffer_stock['tris molarity']
            * mexperiment['tris buffer / L']
            / mexperiment['total vol / L']
        )

        if mexperiment['Sample number'] in range(25, 37):
            salt_molarity = (
                magnesium_chloride_stock['molarity / M']
                * mexperiment['salt stock / L']
                / mexperiment['total vol / L']
            )
        else:
            salt_molarity = (
                calcium_chloride_stock['molarity / M']
                * mexperiment['salt stock / L']
                / mexperiment['total vol / L']
            )

        salt_moles = salt_molarity * mexperiment['total vol / L']

        mexperiment['salt stock molarity / M'] = salt_molarity
        mexperiment['salt moles'] = salt_moles

    metal_real_experiments_rounded = [
        {key: round(x, 5) for key, x in mexperiment.items()}
        for mexperiment in metal_real_experiments
    ]

    metal_output = mo.ui.table(
        data=metal_real_experiments_rounded,
        label='Experiment Data',
    )
    return metal_output, metal_real_experiments


@app.cell
def _(
    adjust_peak_values,
    calculate_ppm_shift,
    data_dir,
    experiment_count,
    experiment_number,
    extract_peak_values,
    extract_sr,
    get_experiment_directories,
    sr_values,
):
    experiment_dir_chelation = (
        '20250811_cit_ca_mg_cit_titr'  # Not fetched it yet
    )

    chelation_experiments = get_experiment_directories(
        data_dir, experiment_dir_chelation, experiment_count
    )

    chelation_experiments[16] = f'{chelation_experiments[16]}_rep'
    chelation_experiments[21] = f'{chelation_experiments[21]}_rep'

    def _():
        chelation_sr_values, chelation_peak_values = [], []
        for experiment in chelation_experiments:
            chelation_sr_values.append(
                extract_sr(experiment, experiment_number, data_dir)
            )
            chelation_peak_values.append(
                extract_peak_values(experiment, experiment_number, data_dir)
            )

        ppm_shift = []

        for sr in sr_values:
            ppm_shift.append(calculate_ppm_shift(sr, frequency=600.5))

        chelation_peak_values = adjust_peak_values(
            chelation_peak_values, ppm_shift
        )

        return chelation_sr_values, chelation_peak_values

    chelation_sr_values, chelation_peak_values = _()
    return (
        chelation_experiments,
        chelation_peak_values,
        experiment_dir_chelation,
    )


@app.cell(hide_code=True)
def _(chelationfig, mo):
    mo.md(rf"""{mo.as_html(chelationfig)}""")
    return


@app.cell
def _(
    bruker_fft,
    chelation_experiments,
    data_dir,
    experiment_number,
    math,
    plt,
):
    def _():
        # Create a new figure
        ngfig = plt.figure(figsize=(20, 15))

        # Calculate the number of rows and columns for subplots
        n = len(chelation_experiments)
        rows = round(math.sqrt(n))
        cols = round(math.ceil(n / rows))

        for idx, experiment in enumerate(chelation_experiments):
            ppmscale, data = bruker_fft(
                data_dir=data_dir,
                experiment=experiment,
                experiment_number=experiment_number,
            )

            ax = ngfig.add_subplot(rows, cols, idx + 1)
            ax.plot(
                ppmscale[20500:21300], data[20500:21300]
            )  # Adjust the range as needed
            # ax.plot(data)
            if idx in range(0, 12):
                ax.set_title(
                    f'Magnesium Chelation Experiment {idx + 1}', fontsize=14
                )
            else:
                ax.set_title(
                    f'Calcium Chelation Experiment {idx - 11}', fontsize=14
                )
            ax.set_xlabel(
                'Data Points', fontsize=12
            )  # Replace with actual x-axis label if needed
            ax.set_ylabel('Magnitude', fontsize=12)     # Magnitude of NMR data
            ax.grid(True)  # Add grid lines for better readability
            ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.suptitle(
            'Chelation Experiments Overview', fontsize=16, y=1.02
        )  # Main title for the figure
        plt.savefig('figs/NMR.svg')
        return plt.gca()

    chelationfig = _()
    return (chelationfig,)


@app.cell
def _(
    bruker_fft,
    data_dir,
    experiment_count,
    experiment_dir_chelation,
    experiment_number,
    get_experiment_directories,
    mo,
    plt,
):
    def _():
        # plt.figure(figsize=(15, 5))

        chelation_experiments = get_experiment_directories(
            data_dir, experiment_dir_chelation, experiment_count
        )

        ppmscale, data = bruker_fft(
            data_dir=data_dir,
            experiment=chelation_experiments[0],
            experiment_number=experiment_number,
        )

        # plt.plot(data[19000:22000])
        plt.plot(ppmscale, data)
        plt.title(f'NMR Experiment', fontsize=14)
        plt.xlabel(
            'Chemical Shift / PPM', fontsize=12
        )  # Replace with actual x-axis label if needed
        plt.ylabel('Magnitude', fontsize=12)     # Magnitude of NMR data
        plt.grid(True)  # Add grid lines for better readability

        plt.legend()

        plt.tight_layout()  # Adjust layout to prevent overlap

        # return plt.gca()

        return plt.gcf()

    singlechelationfig = _()

    mo.mpl.interactive(singlechelationfig)
    return


@app.cell(hide_code=True)
def _(chelation_extra_fig, chelation_fig, mo):
    mo.md(
        rf"""
    ## Chelation Experiment Results

    ### Initial Relationships

    {mo.as_html(chelation_fig)}

    Magnesium appears sigmoidal, calcium appears exponential and reverse exponential.

    ### Extra Relationships

    {mo.as_html(chelation_extra_fig)}

    Magnesium is sigmoidal to j coupling, and calcium is
    """
    )
    return


@app.cell
def _(chelation_peak_values, metal_real_experiments, plt):
    chelation_peak_values_no_intensities = [
        [float(x[0]) for x in y] for y in chelation_peak_values
    ]   # Chemical shift of each peak in citrate [experiments, 4(ppm)]

    def _():
        magnesium_peaks, calcium_peaks = [], []
        for idx, mexperiment in enumerate(metal_real_experiments):
            if mexperiment['Sample number'] in range(25, 37):
                magnesium_peaks.append(
                    chelation_peak_values_no_intensities[idx]
                )
                continue

            calcium_peaks.append(chelation_peak_values_no_intensities[idx])

        return magnesium_peaks, calcium_peaks

    magnesium_peaks, calcium_peaks = _()
    magnesium_percentages = [
        exp['salt stock molarity / M']
        for exp in metal_real_experiments
        if exp.get('Sample number') in range(25, 37)
    ]

    calcium_percentages = [
        exp['salt stock molarity / M']
        for exp in metal_real_experiments
        if exp.get('Sample number') > 36
    ]

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(
        magnesium_percentages,
        magnesium_peaks,
        marker='o',
        linestyle='-',
        linewidth=2,
    )

    plt.title('Magnesium Molarity vs Citrate Chemical Shift ', fontsize=14)
    plt.xlabel('Magnesium Molarity', fontsize=12)
    plt.ylabel('Chemical Shift / ppm', fontsize=12)
    # plt.legend(['Proton A split 1', 'Proton A split 2', 'Proton B split 1', 'Proton B split 2'])

    plt.subplot(2, 1, 2)
    plt.plot(
        calcium_percentages,
        calcium_peaks,
        marker='o',
        linestyle='-',
        linewidth=2,
    )
    plt.title('Calcium Molarity vs Citrate Chemical Shift', fontsize=14)
    plt.xlabel('Calcium Molarity / M', fontsize=12)
    plt.ylabel('Chemical Shift / ppm', fontsize=12)
    # plt.legend(['Proton A split 1', 'Proton A split 2', 'Proton B split 1', 'Proton B split 2'])

    # plt.gca().invert_xaxis()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig('figs/chelation.svg')

    chelation_fig = plt.gca()
    return (
        calcium_peaks,
        calcium_percentages,
        chelation_fig,
        chelation_peak_values_no_intensities,
        magnesium_peaks,
        magnesium_percentages,
    )


@app.cell
def _(calcium_peaks, magnesium_peaks):
    magnesium_peak_shifts = [
        ((pea[0] + pea[1]) / 2) - ((pea[2] + pea[3]) / 2)
        for pea in magnesium_peaks
    ]

    magnesium_j_coupling = [
        [pea[0] - pea[1], pea[2] - pea[3]] for pea in magnesium_peaks
    ]

    calcium_j_coupling = [
        [pea[0] - pea[1], pea[2] - pea[3]] for pea in calcium_peaks
    ]
    calcium_peak_shifts = [
        ((pea[0] + pea[1]) / 2) - ((pea[2] + pea[3]) / 2)
        for pea in calcium_peaks
    ]
    return (
        calcium_j_coupling,
        calcium_peak_shifts,
        magnesium_j_coupling,
        magnesium_peak_shifts,
    )


@app.cell
def _(
    calcium_j_coupling,
    calcium_peak_shifts,
    calcium_percentages,
    magnesium_j_coupling,
    magnesium_peak_shifts,
    magnesium_percentages,
    plt,
):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(
        magnesium_percentages,
        magnesium_j_coupling,
        marker='o',
        linestyle='-',
        linewidth=2,
    )

    plt.title('Magnesium Molarity vs magnesium_j_coupling ', fontsize=14)
    plt.xlabel('Magnesium Molarity', fontsize=12)
    plt.ylabel('magnesium_j_coupling', fontsize=12)

    plt.subplot(2, 2, 2)
    plt.plot(
        magnesium_percentages,
        magnesium_peak_shifts,
        marker='o',
        linestyle='-',
        linewidth=2,
    )

    plt.title('Magnesium Molarity vs magnesium_peak_shifts ', fontsize=14)
    plt.xlabel('Magnesium Molarity', fontsize=12)
    plt.ylabel('magnesium_peak_shifts', fontsize=12)

    plt.subplot(2, 2, 3)
    plt.plot(
        calcium_percentages,
        calcium_j_coupling,
        marker='o',
        linestyle='-',
        linewidth=2,
    )

    plt.title('Calcium Molarity vs calcium_j_coupling Shift ', fontsize=14)
    plt.xlabel('Magnesium Molarity', fontsize=12)
    plt.ylabel('calcium_j_coupling', fontsize=12)

    plt.subplot(2, 2, 4)
    plt.plot(
        calcium_percentages,
        calcium_peak_shifts,
        marker='o',
        linestyle='-',
        linewidth=2,
    )

    plt.title('Calcium Molarity vs calcium_peak_shifts ', fontsize=14)
    plt.xlabel('Magnesium Molarity', fontsize=12)
    plt.ylabel('calcium_peak_shifts', fontsize=12)

    # plt.gca().invert_xaxis()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig('figs/chelation-extra.svg')

    chelation_extra_fig = plt.gca()

    print(
        [
            f'x: {x}, y: {y}'
            for x, y in zip(calcium_percentages, calcium_j_coupling)
        ]
    )
    return (chelation_extra_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Properly Fitting the Chelation Graphs

    $\delta_{cit}=f_{0}\delta_{L}+f_{1}\delta_{ML}$

    $\delta_L$ = chemical shift from the free citric acid (from before), $\delta_{ML}$ = chemical shift from the metal complex

    We know the $f_0, f_1, \delta_{L}$. We don't know the $\delta_{ML}$. Then, once we have the constant $\delta_{ML}$ for each peak, we can find the $f_0,f_1$, which should give ratio of complex to free citric acid. This model assumes there is only a single complex.
    """
    )
    return


@app.cell
def _(math, metal_real_experiments):
    # This assumes metals citrate only form a 1:1 complex.
    # kf = [ML]/([M_free][L_free])
    # M_free = [M_tot] - [ML]
    # L_free = [L_tot] - [ML]
    # kf = [ML]/(([M_tot] - [ML])([L_tot] - [ML]))
    # a = b/((c-b)(d-b))
    # b = (-sqrt(a^2 (c - d)^2 + 2 a (c + d) + 1) + a (c + d) + 1)/(2 a) and a!=0 and sqrt(a^2 (c - d)^2 + 2 a (c + d) + 1)!=a (c + d) + 1 (https://www.wolframalpha.com/input?i=solve+for+b%3A+a+%3D+b%2F%28%28c-b%29%28d-b%29%29)

    def solve_for_b(a, c, d, tol=1e-12):
        # special-case a ≈ 0: linear limit from the quadratic reduces to b = 0 (physically)
        if abs(a) < tol:
            return 0.0

        disc = a * a * (c - d) ** 2 + 2 * a * (c + d) + 1.0
        if disc < -tol:
            return 0.0

        disc = max(disc, 0.0)
        numerator = (
            a * (c + d) + 1.0 - math.sqrt(disc)
        )   # minus branch (physical)
        b = numerator / (2.0 * a)

        # apply physical acceptance tests
        if b < -tol or b > min(c, d) + tol:
            return 0.0
        if abs((c - b) * (d - b)) <= tol:
            return 0.0
        return max(b, 0.0)

    k1 = 3.011e3   # Formation constant of calcium citrate, k11, from source
    k2 = 2.19e3   # Formation constant of magnesium citrate, from https://www.sciencedirect.com/science/article/abs/pii/0003269774903911, incorrect temperature (37C) but I can't find anything at the correct temperature or the values I need to correct

    def _():
        for idx, mexperiment in enumerate(metal_real_experiments):
            if idx < 12:
                a = k2
            else:
                a = k1

            c = mexperiment['salt stock molarity / M']
            d = mexperiment['citric acid molarity / M']

            if c == 0:
                b = 0.0
            else:
                b = solve_for_b(a, c, d)

            mexperiment['metal ligand complex molarity / M'] = b

    _()

    continueaaa = True
    return (continueaaa,)


@app.cell
def _(continueaaa, metal_real_experiments, plt):
    if continueaaa:
        pass
    # Extracting the data
    salt_stock_molarity = [x['salt stock molarity / M'] for x in metal_real_experiments]
    metal_ligand_complex_molarity = [x['metal ligand complex molarity / M'] for x in metal_real_experiments]

    # Combining the data into a list of tuples
    combined_data = list(zip(salt_stock_molarity, metal_ligand_complex_molarity))

    calcium_combined = combined_data[12:]
    magnesium_combined = combined_data[:12]

    # Sorting the combined data by salt stock molarity
    sorted_calcium = sorted(calcium_combined, key=lambda x: x[0])
    sorted_magnesium = sorted(magnesium_combined, key=lambda x: x[0])

    # Unzipping the sorted data back into two lists
    sorted_calcium_molarity, sorted_calcium_complex_molarity = zip(*sorted_calcium)
    sorted_magnesium_molarity, sorted_magnesium_complex_molarity = zip(*sorted_magnesium)

    # Plotting the sorted data
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.plot(sorted_calcium_molarity, sorted_calcium_complex_molarity)
    plt.xlabel('Salt Stock Molarity (M)')
    plt.ylabel('Metal Ligand Complex Molarity (M)')
    plt.title('Plot of Calcium Metal Ligand Complex Molarity vs. Salt Stock Molarity')

    plt.subplot(1, 2, 2)
    plt.plot(sorted_magnesium_molarity, sorted_magnesium_complex_molarity)
    plt.xlabel('Salt Stock Molarity (M)')
    plt.ylabel('Metal Ligand Complex Molarity (M)')
    plt.title('Plot of Magnesium Metal Ligand Complex Molarity vs. Salt Stock Molarity')

    plt.tight_layout()

    plt.show()
    return


@app.cell
def _(
    all_deltas,
    calcium_peaks,
    citricacid,
    continueaaa,
    find_peaks,
    magnesium_peaks,
    metal_real_experiments,
    np,
):
    # We know f0 and f1 and L, but not ML
    # Hopefully, it's a constant
    # shift=f0*L+f1*ML
    # shift-f0*L=f1*ML
    # (shift-f0*l)/f1=ML
    # a = (b-cd)/e

    if continueaaa:
        pass

    assumed_ratios = citricacid.alpha([7.2])

    assumed_shifts = find_peaks(all_deltas=all_deltas, ratios=assumed_ratios)

    def _():
        all_peaks = magnesium_peaks + calcium_peaks
        transposed_peaks = [list(x) for x in zip(*all_peaks)]

        deltas = []
        for peaks, assumedpeak in zip(transposed_peaks, assumed_shifts):
            variables = []
            for (idx, mexperiment), peak in zip(
                enumerate(metal_real_experiments), peaks
            ):
                b = peak   # observed shift
                d = assumedpeak   # shift based on speciation

                if (
                    mexperiment['metal ligand complex molarity / M']
                    is not None
                ):
                    complex = mexperiment['metal ligand complex molarity / M']
                    total = mexperiment['citric acid molarity / M']
                    free_citrate = total - complex

                    c = free_citrate / total
                    # ratio of ligand to metal ligand
                    e = complex / total
                    # ratio of metal ligand to ligand
                else:
                    c = 0
                    e = 1

                variables.append([b, c, d, e])

            variables = [variables[:12], variables[12:]]
            for vars in variables:
                b = np.array([v[0] for v in vars])
                c = np.array([v[1] for v in vars])
                d = np.array([v[2] for v in vars])
                e = np.array([v[3] for v in vars])

                rhs = b - c * d

                a, *_ = np.linalg.lstsq(e.reshape(-1, 1), rhs, rcond=None)

                deltas.append(a[0])

        calcium_deltas = [deltas[1], deltas[3], deltas[5], deltas[7]]
        magnesium_deltas = [deltas[0], deltas[2], deltas[4], deltas[6]]

        return calcium_deltas, magnesium_deltas

    calcium_deltas, magnesium_deltas = _()

    print(calcium_deltas)
    return assumed_shifts, calcium_deltas, magnesium_deltas


@app.cell
def _(
    assumed_shifts,
    calcium_deltas,
    calcium_peaks,
    magnesium_deltas,
    magnesium_peaks,
    metal_real_experiments,
    np,
    plt,
):
    # Try and reverse calculate
    # Now, we know l, and the ML constant, and shift, but we don't know f0 and f1, only that they have to add up to 1.0
    # (shift-f0*l)/f1=ML
    # f0 + f1 = 1.0
    # f0 = 1 - f1
    # (shift-(1 - f1)*l)/f1=ML
    # a = (b-(1-cd)/c
    # a = (b-d(-c+1))/c
    # a = (b+cd-d)/c
    # ac = b+cd-d
    # ac-cd = b-d
    # c(a-d) = b-d
    # c = (b-d)/(a-d)

    def _():
        from scipy.optimize import lsq_linear

        all_peaks = magnesium_peaks + calcium_peaks

        innerdict = {
            'e': [],
            'c': [],
            'real_e': [],
            'real_c': [],
        }

        dict = {
            'magnesium': {k: v.copy() for k, v in innerdict.items()},
            'calcium': {k: v.copy() for k, v in innerdict.items()},
        }

        for (idx, mexperiment), peaks in zip(
            enumerate(metal_real_experiments), all_peaks
        ):
            if idx < 12:
                a = np.array(magnesium_deltas)
                key = 'magnesium'
            else:
                a = np.array(calcium_deltas)   # Deltas array
                key = 'calcium'

            b = np.array(peaks)
            d = np.array(assumed_shifts)

            A = (a - d)[:, None]
            y = b - d
            res = lsq_linear(A, y, bounds=(0.0, 1.0))

            c = res.x[0]   # ratio of metal ligand to ligand
            e = 1.0 - c   # ratio of ligand to metal ligand

            if mexperiment['metal ligand complex molarity / M'] is not None:
                complex = mexperiment['metal ligand complex molarity / M']
                total = mexperiment['citric acid molarity / M']
                free_citrate = total - complex

                real_e = (
                    free_citrate / total
                )   # ratio of ligand to metal ligand
                real_c = complex / total   # ratio of metal ligand to ligand
            else:
                real_e = 1   # ratio of ligand to metal ligand
                real_c = 0   # ratio of metal ligand to ligand

            dict[key]['e'].append(e)
            dict[key]['c'].append(c)
            dict[key]['real_e'].append(real_e)
            dict[key]['real_c'].append(real_c)

        return dict

    chelation_predictions = _()

    def mse(pred, true):
        pred = np.asarray(pred, dtype=float)
        true = np.asarray(true, dtype=float)
        if pred.shape != true.shape:
            raise ValueError('Shapes must match')
        return float(np.sum((true - pred) ** 2))

    errors = {}
    for ion, vals in chelation_predictions.items():
        errors[ion] = {
            'mse_e': mse(vals['e'], vals['real_e']),
            'mse_c': mse(vals['c'], vals['real_c']),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, ion in zip(axes, chelation_predictions):
        vals = chelation_predictions[ion]
        ax.plot(vals['real_e'], vals['e'], marker='o')
        ax.plot(vals['real_c'], vals['c'], marker='s')
        identity = np.linspace(0, 1, 100)
        ax.plot(identity, identity, '--', color='gray')
        ax.set_title(f'{ion.capitalize()}')
        ax.set_xlabel('Calculated')
        ax.set_ylabel('Predicted')
        mse_e = mse(vals['e'], vals['real_e'])
        mse_c = mse(vals['c'], vals['real_c'])
        ax.legend(
            [
                f'e MSE: {round(mse_e,4) if mse_e is not None else "N/A"}',
                f'c MSE: {round(mse_c,4) if mse_c is not None else "N/A"}',
                'Perfect',
            ]
        )
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fitting All Data Together

    The issue with these current models is that they are all considered separate from one another. The speciation model will attempt to explain away metal ligand shifts in terms of speciation, which will in turn mess with the results of the chelation model. Therefore, an integrated model would work better.

    $\delta_{obs, peak}=f_{AH3}\delta_{AH3}+f_{AH2-}\delta_{AH2-}+f_{AH-2}\delta_{aAH-2}+f_{A-3}\delta_{A-3}+f_{MgL}\delta_{MgL}+f_{CaL}\delta_{CaL}$

    $\delta_{obs, peak}=f_{0}\delta_{0}+f_{1}\delta_{1}+f_{2}\delta_{2}+f_{3}\delta_{3}+f_{4}\delta_{4}+f_{5}\delta_{5}$

    Or maybe:

    $\delta_{obs, peak}=f_{L}(f_{AH3}\delta_{AH3}+f_{AH2-}\delta_{AH2-}+f_{AH-2}\delta_{aAH-2}+f_{A-3}\delta_{A-3})+f_{MgL}\delta_{MgL}+f_{CaL}\delta_{CaL}$

    $\delta_{shift}=f_6(f_{0}\delta_{0}+f_{1}\delta_{1}+f_{2}\delta_{2}+f_{3}\delta_{3})+f_{4}\delta_{4}+f_{5}\delta_{5}$
    """
    )
    return


@app.cell
def _(
    chelation_peak_values_no_intensities,
    citricacid,
    continueaaa,
    fracs,
    metal_real_experiments,
    peak_values_no_intensities,
):
    """Concatenate all experiments into an array with only the required values"""

    if continueaaa:
        pass

    def _():
        na_experiments = [{
            'f0': a,
            'f1': b,
            'f2': c,
            'f3': d,
            'f4': 0.0,
            'f5': 0.0,
            'f6': 1.0,
        }
        for a, b, c, d in fracs
        ]

        for idx, experiment in enumerate(na_experiments):
            experiment['peaks'] = peak_values_no_intensities[idx]

        assumed_fracs = citricacid.alpha([7.2])

        mgca_experiments = []
        for idx, mexperiment in enumerate(metal_real_experiments):
            complex = mexperiment['metal ligand complex molarity / M']
            total = mexperiment['citric acid molarity / M']
            free_citrate = total - complex

            c = free_citrate / total
            # ratio of ligand to metal ligand
            e = complex / total
            # ratio of metal ligand to ligand

            if idx < 12:
                f4 = e
                f5 = 0.0
            else:
                f4 = 0.0
                f5 = e

            mgca_experiments.append({
                'f0': assumed_fracs[0],
                'f1': assumed_fracs[1],
                'f2': assumed_fracs[2],
                'f3': assumed_fracs[3],
                'f4': f4,
                'f5': f5,
                'f6': c,
            })

        for idx, experiment in enumerate(mgca_experiments):
            experiment['peaks'] = chelation_peak_values_no_intensities[idx]

        all_experiments = na_experiments + mgca_experiments

        return all_experiments

    all_experiments = _()
    return (all_experiments,)


@app.cell
def _(all_experiments, np):
    """
    Calculate integrated model deltas using nonlinear least squares
    $\delta_{shift}=f_6(f_{0}\delta_{0}+f_{1}\delta_{1}+f_{2}\delta_{2}+f_{3}\delta_{3})+f_{4}\delta_{4}+f_{5}\delta_{5}$
    """
    from scipy.optimize import least_squares

    def _():
        # Each peak is calculated separately, so we have to extract the peaks
        all_peaks = [x['peaks'] for x in all_experiments]
        transposed_peaks = [list(x) for x in zip(*all_peaks)]

        # Extract f values for all experiments
        fs_matrix = []
        for experiment in all_experiments:
            fs_matrix.append([
                experiment['f0'], experiment['f1'], experiment['f2'], 
                experiment['f3'], experiment['f4'], experiment['f5'], experiment['f6']
            ])
        fs_matrix = np.array(fs_matrix)

        deltas = []

        for peak_idx, peaks in enumerate(transposed_peaks):
            peaks = np.array(peaks)

            def residuals(deltas_peak):
                """
                deltas_peak: [δ0, δ1, δ2, δ3, δ4, δ5] for this peak
                """
                predicted_shifts = []

                for exp_idx in range(len(all_experiments)):
                    f0, f1, f2, f3, f4, f5, f6 = fs_matrix[exp_idx]

                    # Model: δ_shift = f6(f0*δ0 + f1*δ1 + f2*δ2 + f3*δ3) + f4*δ4 + f5*δ5
                    speciation_term = f0 * deltas_peak[0] + f1 * deltas_peak[1] + f2 * deltas_peak[2] + f3 * deltas_peak[3]
                    chelation_term = f4 * deltas_peak[4] + f5 * deltas_peak[5]

                    predicted_shift = f6 * speciation_term + chelation_term
                    predicted_shifts.append(predicted_shift)

                predicted_shifts = np.array(predicted_shifts)
                return predicted_shifts - peaks

            # Initial guess for deltas - use average peak position
            x0 = np.full(6, np.mean(peaks))

            # Solve nonlinear least squares
            result = least_squares(residuals, x0, method='trf')

            if not result.success:
                print(f"Warning: optimization failed for peak {peak_idx}: {result.message}")

            deltas.append(result.x)

        return np.array(deltas)

    integrated_deltas = _()
    print("Integrated deltas shape:", integrated_deltas.shape)
    print("Integrated deltas:")
    print(integrated_deltas)
    return integrated_deltas, least_squares


@app.cell
def _(all_deltas, calcium_deltas, magnesium_deltas, np):
    """Alternatively, create the integrated deltas from the previous three models"""

    def _():
        out = []
        for deltas, mg, ca in zip(all_deltas, magnesium_deltas, calcium_deltas):
            test = [*deltas]
            test.append(mg)
            test.append(ca)
            out.append(test)

        return np.array(out)

    alternative_integrated_deltas = _()

    print(alternative_integrated_deltas)
    return


@app.cell
def _(all_experiments, integrated_deltas, least_squares, np):
    """
    Calculate f values from integrated deltas and chemical shifts
    $\delta_{shift}=f_6(f_{0}\delta_{0}+f_{1}\delta_{1}+f_{2}\delta_{2}+f_{3}\delta_{3})+f_{4}\delta_{4}+f_{5}\delta_{5}$
    """

    def find_f_integrated(all_deltas, shifts, penalty_factor=1e3):
        """
        all_deltas: shape (4,6)  (4 peaks x 6 deltas)
        shifts: shape (4,) - observed chemical shifts for 4 peaks
        Returns: optimized f values (f0..f6)
        """

        def residuals(f):
            f0_to_3 = f[:4]  # Speciation fractions
            f4_to_5 = f[4:6]  # Metal complex fractions
            f6 = f[6]        # Free ligand fraction

            # Model: δ_shift = f6(f0*δ0 + f1*δ1 + f2*δ2 + f3*δ3) + f4*δ4 + f5*δ5
            speciation_contributions = all_deltas[:, :4] @ f0_to_3  # Shape: (4,)
            chelation_contributions = all_deltas[:, 4:6] @ f4_to_5  # Shape: (4,)

            predicted_shifts = f6 * speciation_contributions + chelation_contributions

            # Main residuals
            main_residuals = predicted_shifts - shifts

            # Constraint penalties
            # Constraint 1: f0 + f1 + f2 + f3 should sum to 1 (speciation fractions)
            speciation_constraint = penalty_factor * (np.sum(f0_to_3) - 1.0)**2

            # Constraint 2: f4 + f5 + f6 should sum to 1 (total ligand fractions)
            total_ligand_constraint = penalty_factor * (np.sum(f[4:7]) - 1.0)**2

            # Return residuals with penalty terms
            return np.concatenate([main_residuals, [speciation_constraint, total_ligand_constraint]])

        # Bounds: all f_i between 0 and 1
        bounds_lower = np.zeros(7)
        bounds_upper = np.ones(7)

        # Better initial guess based on expected behavior
        x0 = np.array([0.25, 0.25, 0.25, 0.25, 0.1, 0.1, 0.8])  # More realistic starting point

        # Run least squares optimization
        res = least_squares(
            residuals,
            x0,
            bounds=(bounds_lower, bounds_upper),
            method='trf'
        )

        if not res.success:
            print("Warning: optimization did not converge:", res.message)

        return res.x

    def _():
        all_peaks = [x['peaks'] for x in all_experiments]

        fs = []
        for peaks in all_peaks:
            # Convert peaks to numpy array and ensure correct shape
            peaks_array = np.array(peaks)
            if len(peaks_array) != 4:
                print(f"Warning: Expected 4 peaks, got {len(peaks_array)}")
                continue

            f_values = find_f_integrated(integrated_deltas, peaks_array)
            fs.append(f_values)

        return fs

    integrated_predictions = _()

    print("Integrated predictions shape:", len(integrated_predictions))
    if integrated_predictions:
        print("Sample prediction:", integrated_predictions[0])
    return (integrated_predictions,)


@app.cell
def _(all_experiments, integrated_predictions, plt):
    def _():
        fs = integrated_predictions[24:]

        for i in range(len(fs)):
            fs[i] = fs[i]
            # fs[i] = fs[i][4:-2]

        plt.plot(fs)
        plt.legend([*all_experiments[0].keys()])

        plt.show()

    #     return plt.gca()

    # tempfig = _()

    # plt.show()

    _()
    return


@app.cell
def _(all_experiments, plt):
    def _():
        values = []
        for experiment in all_experiments:
            vals = [*experiment.values()]
            vals = vals[:-1]
            values.append(vals)

        plt.plot(values[24:])
        plt.legend([*all_experiments[0].keys()])
        plt.show()

    _()
    return


@app.cell
def _(all_experiments, integrated_predictions, np):
    def _():
        errors = []
        for experiment, prediction in zip(all_experiments, integrated_predictions):
            vals = [*experiment.values()]
            vals.pop()

            error = 0
            for val, f in zip(vals, prediction):
                error += (val-f)**2
            errors.append(error)

        print(np.mean(errors))

    _()
    return


@app.cell
def _(all_experiments, citricacid, integrated_deltas, np):
    """
    Generate additional synthetic data from previous models
    ds=f6(f0d0+f1d1+f2d2+f3d3)+f4d4+f5d5
    Just a straight forward calculation (though not entirely great, since we're depending on the intrinisc chemical shifts from the 10% MSE model. Probably good for noise though.)
    """

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_peaks = torch.tensor([x['peaks'] for x in all_experiments], dtype=torch.float32).to(device)
    test_vals = torch.tensor([list(x.values())[:-1] for x in all_experiments], dtype=torch.float32).to(device)


    train_peaks = []
    train_vals = []

    # Comment out to remove real samples from training
    # train_peaks = [x['peaks'] for x in all_experiments]
    # train_vals = [list(x.values())[:-1] for x in all_experiments]

    # Comment out to remove synthetic samples
    for i in range(10000):
        pH = np.random.default_rng().uniform(3, 7)
        random_speciation_fracs = citricacid.alpha(pH)

        rng = np.random.default_rng()

        if i < 1000:
            random_complex_fracs = [0.0, 0.0, 1.0]
        else:
            random_complex_fracs = rng.dirichlet([1, 1, 1])

        temp_peaks = []
        for deltas in integrated_deltas:
            dl = random_complex_fracs[2] * sum(deltas[i] * random_speciation_fracs[i] for i in range(4))
            dmg = deltas[4] * random_complex_fracs[0]
            dca = deltas[5] * random_complex_fracs[1]

            ds = dl + dmg + dca

            temp_peaks.append(ds)

        temp_peaks = [x + np.random.normal(0, 0.01) for x in temp_peaks]

        train_peaks.append(temp_peaks)

        train_vals.append([*random_speciation_fracs, *random_complex_fracs])

    print(len(train_peaks))
    print(len(train_vals))
    return device, test_peaks, test_vals, torch, train_peaks, train_vals


@app.cell
def _():
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    return F, nn, optim


@app.cell
def _(
    F,
    TinyTransformer,
    device,
    nn,
    np,
    optim,
    torch,
    train_peaks,
    train_vals,
):
    ### Try a neural network?

    class ConstrainedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 7),
            )

        def forward(self, x):
            x = self.net(x)
            first4 = F.softmax(x[:, :4], dim=1)   # first 4 constrained
            last3 = F.softmax(x[:, 4:], dim=1)    # last 3 constrained
            return torch.cat([first4, last3], dim=1)

    model = TinyTransformer(4).to(device)

    X = torch.tensor(train_peaks, dtype=torch.float32).to(device)  # (samples, 4)
    y = torch.tensor(train_vals, dtype=torch.float32).to(device)  # (samples, 7)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = np.inf
    loss_diff = 0.000001
    full_count = 0
    count = 0
    while True:
    # for epoch in range(epochs):
        model.train()  # ensure model is in training mode

        optimizer.zero_grad()      # reset gradients
        outputs = model(X)         # forward pass
        loss = criterion(outputs, y)  # compute loss
        loss.backward()            # backpropagate
        optimizer.step()           # update weights

        if loss.item() < best_loss - loss_diff:
            best_loss = loss.item()

            count = 0

        count += 1
        full_count += 1

        if count > 1000:
            break

    print(f'{full_count} epochs with {best_loss} loss')
    return criterion, model


@app.cell
def _(all_experiments, criterion, model, plt, test_peaks, test_vals, torch):
    def _():
        model.eval()
        with torch.no_grad():
            slicer = slice(None) # All
            # slicer = slice(0, 24)
            # slicer = slice(24, 48)

            predictions = model(test_peaks)
            loss = criterion(predictions, test_vals)  # compute loss
            predictions = predictions.cpu()
            print(f'Test loss: {loss.item()}')

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(predictions[slicer])
            plt.legend([*all_experiments[0].keys()])
            plt.title('Predictions')


            plt.subplot(1, 2, 2)
            plt.plot([list(x.values())[:-1] for x in all_experiments][slicer])
            plt.legend([*all_experiments[0].keys()])
            plt.title('real')

            plt.show()

            return predictions

    predictions = _()
    return (predictions,)


@app.cell
def _(all_experiments, np, predictions):
    def _():
        errors = []
        for experiment, prediction in zip(all_experiments, predictions):
            vals = [*experiment.values()]
            vals.pop()

            error = 0
            for val, f in zip(vals, prediction):
                error += (val-f)**2
            errors.append(error)

        print(np.mean(errors))

    _()
    return


@app.cell
def _(all_experiments, metal_real_experiments, plt, predictions):
    def _():
        mg = [0] * 24
        ca = [0] * 24
        for experiment in metal_real_experiments:
            if experiment['Sample number'] > 36:
                ca.append(experiment['salt stock molarity / M'])
                mg.append(0)
            else:
                mg.append(experiment['salt stock molarity / M'])
                ca.append(0)
        print(len(mg))
        print(len(ca))

        residuals = []
        for experiment, prediction in zip(all_experiments, predictions):
            vals = [*experiment.values()]
            vals.pop()

            r = []
            for val, f in zip (vals, prediction):
                r.append(abs(val-f))
            residuals.append(r)

        plt.plot(mg, residuals)
        plt.legend([*all_experiments[0].keys()])
        plt.show()

    _()
    return


@app.cell
def _(
    bruker_fft,
    chelation_experiments,
    data_dir,
    experiment_number,
    experiments,
    np,
):
    def _():
        all_experiments_fft = []

        for idx, experiment in enumerate(experiments):
            all_experiments_fft.append(bruker_fft(
                data_dir=data_dir,
                experiment=experiment,
                experiment_number=experiment_number,
            ))

        for idx, experiment in enumerate(chelation_experiments):
            all_experiments_fft.append(bruker_fft(
                data_dir=data_dir,
                experiment=experiment,
                experiment_number=experiment_number,
            ))

        all_experiments_fft = np.array(all_experiments_fft)

        sliced_experiments = []
        for spectrum in all_experiments_fft:
            lower_bound, upper_bound = 2.4, 2.8

            x = spectrum[0]

            mask = (x >= lower_bound) & (x <= upper_bound)
            sliced_experiments.append(spectrum[:, mask])

        length = max([len(x[0]) for x in sliced_experiments])

        for i, spectrum in enumerate(sliced_experiments):
            diff = len(spectrum[0]) - length
            if diff > 0: # Too long
                spectrum = spectrum[:, :-diff]
            if diff < 0: #Too short
                # Pad (2, x) to (2, x + diff)
                spectrum = np.pad(spectrum, ((0, 0), (0, -diff)), mode='constant', constant_values=0)
            sliced_experiments[i] = spectrum

        lengths = set([len(x[0]) for x in sliced_experiments])

        if len(lengths) > 1:
            raise ValueError('Lengths is greater than 1')

        return sliced_experiments

    all_experiments_fft = _()
    return (all_experiments_fft,)


@app.cell
def _(F, math, nn, torch):
    class TinyTransformer(nn.Module):
        def __init__(self, length, seq_len=4, token_dim=None, d_model=64, d_ff=128):
            """
            length: original input vector length
            seq_len: number of sequence tokens to split input into
            token_dim: token dimension; if None, computed as ceil(length/seq_len)
            d_model: internal model dimension (projection size)
            d_ff: feed-forward hidden size
            """
            super().__init__()

            # determine token_dim so that seq_len * token_dim >= length
            if token_dim is None:
                token_dim = math.ceil(length / seq_len)
            self.seq_len = seq_len
            self.token_dim = token_dim
            self.length = length

            # linear projection from token_dim -> d_model
            self.token_proj = nn.Linear(token_dim, d_model)

            # positional embeddings (learned)
            self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

            # single-head self-attention (scaled dot-product)
            self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)  # produce q,k,v
            self.out_proj = nn.Linear(d_model, d_model)

            # small feed-forward
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
            )

            # layer norms
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

            # classification head: pool and map to 7 logits
            self.pool_proj = nn.Linear(d_model, d_model)
            self.classifier = nn.Linear(d_model, 7)

        def _prepare_tokens(self, x):
            # x: (batch, length)
            B = x.shape[0]
            L = self.length
            T = self.seq_len
            D = self.token_dim

            # ensure x has size length; if input length differs, pad or truncate
            if x.shape[1] < L:
                pad = x.new_zeros((B, L - x.shape[1]))
                x = torch.cat([x, pad], dim=1)
            elif x.shape[1] > L:
                x = x[:, :L]

            # reshape into tokens: (batch, seq_len, token_dim)
            x = x.reshape(B, T, D)
            return x

        def _self_attention(self, x):
            # x: (B, T, d_model)
            B, T, DM = x.shape
            qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
            q, k, v = qkv.chunk(3, dim=-1)
            scale = 1.0 / math.sqrt(DM)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, T, T)
            attn_weights = F.softmax(attn_logits, dim=-1)
            out = torch.matmul(attn_weights, v)  # (B, T, d_model)
            out = self.out_proj(out)
            return out

        def forward(self, x):
            # x: (batch, length)
            # 1) prepare tokens
            tokens = self._prepare_tokens(x)                  # (B, seq_len, token_dim)
            tokens = self.token_proj(tokens)                  # (B, seq_len, d_model)
            tokens = tokens + self.pos_emb.unsqueeze(0)       # add pos emb

            # 2) transformer block (one layer)
            y = self.ln1(tokens + self._self_attention(tokens))
            y = self.ln2(y + self.ff(y))

            # 3) simple pooling (mean over seq) and classification
            pooled = y.mean(dim=1)            # (B, d_model)
            pooled = F.relu(self.pool_proj(pooled))
            logits = self.classifier(pooled)  # (B, 7)

            # 4) split into constrained softmax groups like original
            first4 = F.softmax(logits[:, :4], dim=1)
            last3 = F.softmax(logits[:, 4:], dim=1)
            return torch.cat([first4, last3], dim=1)

    return (TinyTransformer,)


@app.cell
def _(F, TinyTransformer, all_experiments_fft, device, nn, np, torch):
    class FullModel(nn.Module):
        def __init__(self, length):
            super().__init__()
            a = length // 2
            b = a // 2
            c = b // 2
            d = c //2

            self.net = nn.Sequential(
                nn.Linear(length, a),
                nn.ReLU(),
                nn.Linear(a, b),
                nn.ReLU(),
                nn.Linear(b, c),
                nn.ReLU(),
                nn.Linear(c, d),
                nn.ReLU(),
                nn.Linear(d, 7)
            )

        def forward(self, x):
            x = self.net(x)
            first4 = F.softmax(x[:, :4], dim=1)   # first 4 constrained
            last3 = F.softmax(x[:, 4:], dim=1)    # last 3 constrained
            return torch.cat([first4, last3], dim=1)

    temp = [np.concatenate(x) for x in all_experiments_fft]

    all_spectra = torch.tensor(temp, dtype=torch.float32).to(device)

    SpectraModel = TinyTransformer(max([len(x) for x in temp])).to(device)
    return SpectraModel, all_spectra


@app.cell
def _(
    SpectraModel,
    all_experiments,
    all_spectra,
    criterion,
    device,
    np,
    optim,
    torch,
):
    def _():
        count = 0
        full_count = 0
        best_loss = np.inf
        optimizer = optim.Adam(SpectraModel.parameters(), lr=1e-3)

        temp_vals = torch.tensor([list(x.values())[:-1] for x in all_experiments], dtype=torch.float32).to(device)
    
        while True:
        # for epoch in range(epochs):
            SpectraModel.train()  # ensure model is in training mode

            optimizer.zero_grad()      # reset gradients
            outputs = SpectraModel(all_spectra)         # forward pass
            loss = criterion(outputs, temp_vals)  # compute loss
            loss.backward()            # backpropagate
            optimizer.step()           # update weights

            if loss.item() < best_loss: # - loss_diff:
                best_loss = loss.item()

                count = 0

            count += 1
            full_count += 1

            if count > 1000:
                break

        print(best_loss)
        print(full_count)

    _()
    return


@app.cell
def _(
    SpectraModel,
    all_experiments,
    all_spectra,
    criterion,
    plt,
    test_vals,
    torch,
):
    def _():
        SpectraModel.eval()
        with torch.no_grad():
            slicer = slice(None) # All
            # slicer = slice(0, 24)
            slicer = slice(24, 48)

            predictions = SpectraModel(all_spectra)
            loss = criterion(predictions, test_vals)  # compute loss
            predictions = predictions.cpu()
            print(f'Test loss: {loss.item()}')

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(predictions[slicer])
            plt.legend([*all_experiments[0].keys()])
            plt.title('Predictions')


            plt.subplot(1, 2, 2)
            plt.plot([list(x.values())[:-1] for x in all_experiments][slicer])
            plt.legend([*all_experiments[0].keys()])
            plt.title('real')

            plt.show()

            return predictions

    spectrapredictions = _()
    return


if __name__ == "__main__":
    app.run()
