import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from phfork import AcidAq, IonAq, System
    import phfork
    import numpy as np
    return mo, np, phfork


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Experimental Method for Citric Acid Speciation Chemical Shift""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate pKas

    The first step is to find the pKa values of citric acid, so that we know what increment of base to acid ratio will lead to a certain pH. We had a table of buffer pH values, but these had two problems: 

    1) they were at the wrong concentration

    2) they only worked over the buffer range, and we wanted to test outside of this range.
    """
    )
    return


@app.cell
def _():
    from chempy import electrolytes

    # Graph constants
    EPS_R = 78.3   # relative permittivity of water at 25°C/298K
    T = 298  # K
    RHO = 0.997
    B0 = 1.0
    A_CONST = electrolytes.A(eps_r=EPS_R, T=T, rho=RHO, b0=B0)
    trials = 1
    search_molarity = 0.1
    graph_molarity = 0.001
    stock_molarity = 0.01
    stock_volume = 50   # ml
    sample_vol = 0.0006   # l
    acid_mass = 21.01   # g/l, 0.1M
    base_mass = 29.41   # g/1, 0.1M
    acid_molecular_weight = 192.12   # g/mol
    base_molecular_weight = 258.07   # g/mol
    dss_molecular_weight = 224.36   # g/mol
    dss_molarity = 0.001
    rounding = 3

    options = [
        0,
        3.4,
        3.5,
        3.6,
        3.7,
        3.8,
        4,
        4.2,
        4.4,
        4.5,
        4.6,
        4.8,
        5,
        5.2,
        5.4,
        5.5,
        5.6,
        5.7,
        5.9,
        6,
        6.2,
        6.4,
        6.8,
        9,
    ]

    print(len(options))
    return (
        A_CONST,
        acid_mass,
        acid_molecular_weight,
        base_mass,
        base_molecular_weight,
        dss_molarity,
        dss_molecular_weight,
        graph_molarity,
        options,
        rounding,
        sample_vol,
        search_molarity,
        stock_molarity,
        stock_volume,
        trials,
    )


@app.cell
def _(phfork):
    def simulate_ph_graph(pka, conc, charge=0):
        """Function to simulate a ph graph at a concentration from pkas"""
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
    return (simulate_ph_graph,)


@app.cell
def _(simulate_ph_graph):
    def evaluate_pka_error(known_values, search_molarity, trial):
        """Evaluate the mean square error in pH values between the known buffer data and the predicted data from a set of pkas"""
        pka_values = [
            trial.suggest_float('pka1', low=2.0, high=3.5, step=0.001),
            trial.suggest_float('pka2', low=4.0, high=5.5, step=0.001),
            trial.suggest_float('pka3', low=5.5, high=6.5, step=0.001),
        ]

        ratios = simulate_ph_graph(pka=pka_values, conc=search_molarity)

        error = 0.0
        for known in known_values:
            closest_entry = min(
                ratios,
                key=lambda d: abs(d['acid ratio'] - known['acid ratio']),
            )
            error += (known['ph'] - closest_entry['pH']) ** 2

        return error
    return (evaluate_pka_error,)


@app.cell
def _():
    """Import data from bufferdata.csv, convert to dict"""

    import pandas as pd

    imported = pd.read_csv('bufferdata.csv')
    imported = imported.to_dict(orient='split', index=False)

    known_values = []

    for data in imported['data']:
        known_values.append(
            {'ph': data[0], 'acid ratio': data[1], 'base ratio': data[2]}
        )

    print(known_values[0])
    print(len(known_values))
    return known_values, pd


@app.cell
def _(evaluate_pka_error, known_values, search_molarity, trials):
    """Use optuna to search for the pka values that fit the buffer data the best"""

    import optuna
    from functools import partial

    def objective(trial, search_molarity, known_values):
        """Evaluate the pka error"""
        error = evaluate_pka_error(
            trial=trial,
            search_molarity=search_molarity,
            known_values=known_values,
        )
        return error

    study = optuna.create_study(
        direction='minimize',
        study_name='PKA_STUDY',
        storage='sqlite:///PKA_STUDY.db',
        load_if_exists=True,
    )

    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )

    if trials - completed_trials > 0:
        study.optimize(
            partial(
                objective,
                known_values=known_values,
                search_molarity=search_molarity,
            ),
            n_jobs=8,
            callbacks=[
                optuna.study.MaxTrialsCallback(
                    trials, states=(optuna.trial.TrialState.COMPLETE,)
                )
            ],
        )
    return (study,)


@app.cell
def _(study):
    """Get the pkas from trial data"""

    pka = [
        round(study.best_trial.params[x], 5) for x in study.best_trial.params
    ]
    print(pka)
    return (pka,)


@app.cell
def _(A_CONST, np):
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
    return correct_pkas, ionic_strength_from_conc


@app.cell
def _(
    correct_pkas,
    graph_molarity,
    ionic_strength_from_conc,
    pka,
    search_molarity,
):
    """Correct pkas with debye huckel between search molarity (molarity of buffer data) and graph molarity (molarity of the samples)"""

    # Citrate acid-base pairs and charges for pKa steps:
    # H3A (0) ⇌ H2A- (-1)
    # H2A- (-1) ⇌ HA2- (-2)
    # HA2- (-2) ⇌ A3- (-3)
    charge_pairs = [(0, -1), (-1, -2), (-2, -3)]

    # Calculate ionic strengths (crudely approximated)
    I_old = ionic_strength_from_conc(search_molarity)
    I_new = ionic_strength_from_conc(graph_molarity)

    # Correct pKas for change in ionic strength
    corrected_pka = [
        float(x) for x in correct_pkas(pka, I_old, I_new, charge_pairs)
    ]

    print(pka)
    print([round(x, 2) for x in corrected_pka])
    return (corrected_pka,)


@app.cell
def _(corrected_pka, graph_molarity, simulate_ph_graph):
    """Final speciation ratios at graph molarity"""
    ratios = simulate_ph_graph(pka=corrected_pka, conc=graph_molarity)
    print(''.join(f'{x}\n' for x in ratios))
    return (ratios,)


@app.cell(hide_code=True)
def _(corrected_pka, graph_molarity, mo, search_molarity, speciationfig):
    mo.md(
        rf"""
    Our final pKa values, calculated at {search_molarity} and corrected to {graph_molarity}, where: {corrected_pka}.

    With these pKa values, we could generate a speciation graph for citric acid:

    {mo.as_html(speciationfig)}
    """
    )
    return


@app.cell
def _(corrected_pka, graph_molarity, np, phfork):
    import matplotlib.pyplot as plt

    # Sample data for demonstration
    phs = np.linspace(1, 9, 1000)

    # Assuming corrected_pka and graph_molarity are defined
    citricacid = phfork.AcidAq(
        pKa=corrected_pka, charge=0, conc=graph_molarity
    )
    fracs = citricacid.alpha(phs)

    # Create a figure with a specific size
    fig = plt.figure(figsize=(10, 6))

    # Plot the fractions with a color palette
    plt.plot(phs, fracs, linewidth=2)

    # Add a legend with a title
    plt.legend(
        ['H3A', 'H2A-', 'HA2-', 'A3-'], title='Species', loc='upper right'
    )

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set title and labels with larger font sizes
    plt.title('Citric Acid Speciation', fontsize=16)
    plt.xlabel('pH', fontsize=14)
    plt.ylabel('Fraction of Species', fontsize=14)

    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the plot
    plt.tight_layout()
    speciationfig = plt.gca()
    return fig, speciationfig


@app.cell(hide_code=True)
def _(experiment_output, mo):
    mo.md(
        rf"""
    ## Experimental Options

    Our next job then was to decide which values of

    {experiment_output}
    """
    )
    return


@app.cell
def _(options, ratios):
    experiments = []
    for option in options:
        experiments.append(
            min(ratios, key=lambda d: abs(float(d['pH']) - option))
        )
    return (experiments,)


@app.cell(hide_code=True)
def _(mo, stock_output):
    mo.md(
        rf"""
    ## Stock Requirements

    {stock_output}
    """
    )
    return


@app.cell
def _(experiments, mo, rounding, sample_vol, stock_molarity):
    acid_vol = 0
    base_vol = 0

    volumed_experiments = []

    for row in experiments:

        acid_multiplier = float(row['acid ratio']) / 100
        acid_vol_add = acid_multiplier * sample_vol

        base_multiplier = float(row['base ratio']) / 100
        base_vol_add = base_multiplier * sample_vol

        acid_vol += acid_vol_add
        base_vol += base_vol_add

        volumed_experiments.append(
            {
                'pH': row['pH'],
                'acid volume': round(
                    acid_vol_add * 1 / stock_molarity, rounding
                ),
                'base volume': round(
                    base_vol_add * 1 / stock_molarity, rounding
                ),
            }
        )

    experiment_output = mo.ui.table(
        data=volumed_experiments, pagination=True, label='Experiment output'
    )
    return acid_vol, base_vol, experiment_output, volumed_experiments


@app.cell
def _(
    acid_mass,
    acid_molecular_weight,
    acid_vol,
    base_mass,
    base_molecular_weight,
    base_vol,
    dss_molarity,
    dss_molecular_weight,
    rounding,
    stock_molarity,
    stock_volume,
):
    acid_weight = acid_mass * acid_vol
    base_weight = base_mass * base_vol

    stock_acid_weight = (
        stock_molarity * (stock_volume / 1000) * acid_molecular_weight
    )
    stock_base_weight = (
        stock_molarity * (stock_volume / 1000) * base_molecular_weight
    )
    dss_weight = dss_molarity * (stock_volume / 1000) * dss_molecular_weight

    stock_output = f"""
    Requirements for {stock_volume}ml {stock_molarity}M stock solution:

    - Acid weight: {round(stock_acid_weight * 1000, rounding)}mg
    - Base weight: {round(stock_base_weight * 1000, rounding)}mg
    - Water requirements: {stock_volume * 0.95}ml
    - D2O requirements: {stock_volume * 0.05}ml
    - DSS requirements: {dss_weight * 1000}mg
    """
    return (stock_output,)


@app.cell
def _(corrected_pka, fig, pd, pka, stock_output, study, volumed_experiments):
    import operator
    import os

    directory = 'output-graph'

    if not os.path.isdir(directory):
        os.mkdir(directory)

    with open(f'{directory}/pka.txt', 'w') as f:
        f.write(f'Error: {round(study.best_trial.value, 10)}\n')
        f.write(f'Original pKas (0.1 M): {pka}\n')
        f.write(
            f'Corrected pKas (0.001 M): {[round(x, 3) for x in corrected_pka]}\n'
        )

    with open(file=f'{directory}/stocks.txt', mode='w') as f:
        f.writelines(stock_output)

    volumed_experiments.sort(key=operator.itemgetter('pH'))

    out = pd.DataFrame.from_dict(volumed_experiments)
    out.to_csv(path_or_buf=f'{directory}/experiments.csv', index=False)

    fig.savefig(f'{directory}/graph.png')
    return


@app.cell(hide_code=True)
def _(
    better_sample_vol,
    calcium_conc,
    magnesium_conc,
    metal_experiment_output,
    metal_stock_output,
    mo,
    stock_molarity,
    tris_conc,
    tris_vol,
):
    mo.md(
        rf"""
    ## Metal Ion Chelation Experimental Preparation

    - Use 1/10 sample vol {round(better_sample_vol * 0.1*1000, 4)}mL citric acid, which is a molarity of {stock_molarity*0.1}M
    - Calcium chloride
        - 0.4% with spin 5/2, practically negligible. [Source](https://webelements.com/calcium/isotopes.html)
        - Final concentration: {calcium_conc}M. [This](https://doi.org/10.1007/s007750100264) paper recommends 1.5e-2mol.dm-3, or 0.015M, but their values remain relatively constant past about 0.005M. x10 so I only have to add 0.1mL.
        - Final volume: {round(better_sample_vol * 0.1*1000, 4)}mL
    - Magnesium chloride
        - 10% 5/2 spin, within reason, maybe. [Source](https://webelements.com/magnesium/isotopes.html)
        - Final concentration: {magnesium_conc}M. [This](https://doi.org/10.1007/s007750100264) paper states that a 1:1 magnesium:citrate complex occurs at 2.34e-3mol.dm-3, or 0.00234M. 10x for volume.
        - Final volume: {round(better_sample_vol * 0.1*1000, 4)}mL
    - Aluminium chloride
        - 100% isotope with 5/2 spin. Not great. [Source](https://webelements.com/aluminium/isotopes.html)
    - Add {tris_vol*1000}ml of tris buffer. [This](https://doi.org/10.1007/s007750100264) paper recommended a target concentration of 5e-2mol.dm-3, or 0.05M, so double that to {tris_conc}M as we're going to use half the volume as buffer.
    - Make up to {better_sample_vol*1000}mL total with milliq

    ### Stock preparation:
    {metal_stock_output}

    ### Experiment preparation:

    {metal_experiment_output}
    """
    )
    return


@app.cell
def _():
    # Metal experiment constants
    better_sample_vol = 0.0015   # l, Use a bigger volume to give leeway, 1.5mL is the maximum the eppendorf can hold
    citric_sample_vol = better_sample_vol * 0.1   # L, 1/10th
    tris_vol = better_sample_vol * 0.5   # L, 1/2

    calcium_conc = 0.005 * 10   # M
    magnesium_conc = 0.00234 * 10   # M
    tris_conc = 0.05 * 2   # M

    magnesium_chloride_mass = 203.30   # g/mol
    calcium_chloride_mass = 147.02   # g/mol
    tris_mass = 121.14   # g/mol
    hcl_mass = 36.46   # g/mol
    tris_chloride_mass = 157.59   # g/mol

    metal_stock_volume = 5 / 1000   # mL to L
    tris_stock_volume = 50 / 1000   # mL to L

    number_experiments_per_metal = 12   # Count (not including 0)

    d2o_percentage = 0.05   # %
    d2o_density = 1.1044   # g/ml
    return (
        better_sample_vol,
        calcium_chloride_mass,
        calcium_conc,
        citric_sample_vol,
        d2o_density,
        d2o_percentage,
        hcl_mass,
        magnesium_chloride_mass,
        magnesium_conc,
        metal_stock_volume,
        number_experiments_per_metal,
        tris_chloride_mass,
        tris_conc,
        tris_mass,
        tris_stock_volume,
        tris_vol,
    )


@app.cell
def _(
    better_sample_vol,
    citric_sample_vol,
    number_experiments_per_metal,
    pd,
    tris_vol,
):
    metal_experiments = []
    metal_eppendorfs_csv = (
        []
    )   # Blank csv to print and write eppendorf weights into

    # Intialise experiments
    for y in range(0, 2):
        if y == 0:
            salt_stock_name = 'magnesium salt stock / µL'
        elif y == 1:
            salt_stock_name = 'calcium salt stock / µL'

        for i in range(0, number_experiments_per_metal):
            metal_experiments.append(
                {
                    'citric acid stock / µL': citric_sample_vol,
                    'tris buffer stock / µL': round(tris_vol, 6),
                    salt_stock_name: citric_sample_vol
                    / (number_experiments_per_metal - 1)
                    * i,
                }
            )

            metal_eppendorfs_csv.append(
                {
                    'eppendorf base weight / g': None,
                    'post citric acid stock weight / g': None,
                    'post salt stock weight / g': None,
                    'post tris buffer weight / g': None,
                    'post milliq weight / g': None,
                }
            )

    # Make up to better_sample_vol with milliq
    for x in metal_experiments:
        temp = 0
        for y in x.items():
            temp += y[1]
        x['milliq µL'] = round(better_sample_vol - temp, 6)

    # Round everything to µL
    for x in metal_experiments:
        for key, value in x.items():
            x[key] = round(value * 1000 * 1000)

    # Double check volumes
    for x in metal_experiments:
        temp = 0
        for y in x.items():
            temp += y[1]
        if temp != better_sample_vol * 1000 * 1000:
            print('Issue')

    metal_experiments = pd.DataFrame(metal_experiments)
    metal_eppendorfs_csv = pd.DataFrame(metal_eppendorfs_csv)

    # Put milliq at the end for simplicity's sake
    columns = [
        col for col in metal_experiments.columns if col != 'milliq µL'
    ] + ['milliq µL']
    metal_experiments = metal_experiments[columns]

    metal_experiments.index = range(
        25, 25 + len(metal_experiments)
    )   # Start index after the number of experiments actually done
    metal_eppendorfs_csv.index = range(25, 25 + len(metal_eppendorfs_csv))
    return metal_eppendorfs_csv, metal_experiments


@app.cell
def _(
    calcium_chloride_mass,
    calcium_conc,
    d2o_density,
    d2o_percentage,
    hcl_mass,
    magnesium_chloride_mass,
    magnesium_conc,
    metal_experiments,
    metal_stock_volume,
    mo,
    tris_chloride_mass,
    tris_conc,
    tris_mass,
    tris_stock_volume,
):
    ## Stocks
    # conc=moles/L
    # conc*L=moles
    # mass=g/moles
    # mass*moles=g
    # g = mass * (conc * L)

    magnesium_chloride_weight = (
        magnesium_chloride_mass * magnesium_conc * metal_stock_volume
    )
    calcium_chloride_weight = (
        calcium_chloride_mass * calcium_conc * metal_stock_volume
    )

    # This value comes from data for biochemical research, it has a buffer table, this is the amount of HCl you need for the ph. 47ml of 0.1M into 100ml of pH 7.2 buffer. Since our stock is double concentration of what we're going to put into the sample, double it.
    hcl_moles = 0.0047 * 0.1 * 2

    tris_weight = tris_mass * tris_conc * tris_stock_volume
    hcl_weight = hcl_mass * hcl_moles
    tris_chloride_weight = tris_chloride_mass * hcl_moles

    # Convert tris_weight and tris_chloride_weight to moles, remove moles of tris_chloride from tris, convert back to weight
    tris_chloride_moles = tris_chloride_weight / tris_chloride_mass
    tris_moles = tris_weight / tris_mass
    tris_weight_chloride_corrected = (
        tris_moles - tris_chloride_moles
    ) * tris_mass

    metal_stock_output = f"""Citric acid stock: use old citric acid stock, and dilute 1/10 into samples. This puts the range of citrate ions into soluble range of calcium citrate, and means we don't have to use more DSS, and the ratio of DSS to citrate will be identical to our previous experiments.

    Metal stocks:

    - Conical flask/total volume: {metal_stock_volume*1000}mL
    - Magnesium chloride: {round(magnesium_chloride_weight*1000, 5)}mg
    - Calcium chloride: {round(calcium_chloride_weight*1000, 5)}mg
    - D2O: {metal_stock_volume*1000*d2o_percentage}ml/{round(metal_stock_volume*1000*d2o_percentage*d2o_density, 5)}g
    - H2O: {metal_stock_volume*1000*(1-d2o_percentage)}ml/g


    Tris buffer stock:

    Using the ratio of chloride moles to tris moles found in Data for Biochemical Research, Third Edition, for buffer pH 7.20, as at this pH tris does not interact strongly with calcium ions (found from [This](https://doi.org/10.1007/s007750100264) publication). Tris peaks found [Here](https://www.chemicalbook.com/SpectrumEN_77-86-1_1HNMR.htm).

    - Conical flask/total volume: {tris_stock_volume*1000}mL
    - Tris: {round(tris_weight*1000, 5)}mg if HCl OR {round(tris_weight_chloride_corrected*1000, 5)}mg if tris chloride
    - HCl: {round(hcl_weight*1000, 5)}mg OR tris chloride: {round(tris_chloride_weight*1000, 5)}mg
    - D2O: {tris_stock_volume*1000*d2o_percentage}ml/{round(tris_stock_volume*1000*d2o_percentage*d2o_density, 5)}g
    - H2O: {tris_stock_volume*1000*(1-d2o_percentage)}ml and identical g
    """

    metal_stock_output_csv = f"""
    Metal stocks:

    - Conical flask/total volume: {metal_stock_volume*1000}mL
    - Magnesium chloride: {round(magnesium_chloride_weight*1000, 5)}mg
    - Calcium chloride: {round(calcium_chloride_weight*1000, 5)}mg
    - D2O: {metal_stock_volume*1000*d2o_percentage}ml/{round(metal_stock_volume*1000*d2o_percentage*d2o_density, 5)}g
    - H2O: {metal_stock_volume*1000*(1-d2o_percentage)}ml/g


    Tris buffer stock:

    - Conical flask/total volume: {tris_stock_volume*1000}mL
    - Tris: {round(tris_weight*1000, 5)}mg if HCl OR {round(tris_weight_chloride_corrected*1000, 5)}mg if tris chloride
    - HCl: {round(hcl_weight*1000, 5)}mg OR tris chloride: {round(tris_chloride_weight*1000, 5)}mg
    - D2O: {tris_stock_volume*1000*d2o_percentage}ml/{round(tris_stock_volume*1000*d2o_percentage*d2o_density, 5)}g
    - H2O: {tris_stock_volume*1000*(1-d2o_percentage)}ml and identical g
    """

    metal_experiment_output = mo.ui.table(
        data=metal_experiments, pagination=True
    )

    print(
        (tris_weight + hcl_weight)
        - (tris_weight_chloride_corrected + tris_chloride_weight)
    )

    print(hcl_moles - tris_chloride_moles)
    return metal_experiment_output, metal_stock_output, metal_stock_output_csv


@app.cell
def _(metal_eppendorfs_csv, metal_experiments, metal_stock_output_csv):
    """Write to file"""

    def _():
        metal_experiments.to_csv('metal_experiments.csv', index=True)
        with open('metal_experiments.csv', 'a') as f:
            f.writelines(metal_stock_output_csv)

        metal_eppendorfs_csv.to_csv(
            'experimental/metal_eppendorfs_blank.csv', index=True
        )

    _()
    return


if __name__ == "__main__":
    app.run()
