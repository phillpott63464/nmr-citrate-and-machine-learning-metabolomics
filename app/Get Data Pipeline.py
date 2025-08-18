import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```
    (base) sp970@sp970-HP-Z440-Workstation:~/sam/data/sp/20250811_cit_nacit_titr_1/3$ grep 600 acqus
    ##$BF1= 600.05
    ##$BF2= 600.05
    ##$BF3= 600.05
    ##$BF4= 600.05
    ##$BF5= 600.05
    ##$BF6= 600.05
    ##$BF7= 600.05
    ##$BF8= 600.05
    ##$FRQLO3= 1246008.49005003
    100000 600 0 0 0 0 0 180 100000 7.89 0 1000 0 0 20000 0 0 0 0 0 0 2400
    2000 2200 1400 1700 200 0 80000 46600 10000 100000 80000 1600 0 0 0 1300
    ##$PROBHD= <Z124831_0001 (CP QCI 600 S3 H&F-P/C&Co/N-D-05 Z)>
    ##$SFO1= 600.05281963495
    ##$SFO2= 600.05281511
    ##$SFO3= 600.05281511
    ##$SFO4= 600.05281511
    ##$SFO5= 600.05281511
    ##$SFO6= 600.05281511
    ##$SFO7= 600.05281511
    ##$SFO8= 600.05281511

    ```

    SF01 - important

    ```
    (base) sp970@sp970-HP-Z440-Workstation:~/sam/data/sp/20250811_cit_nacit_titr_1/3$ grep 600 pdata/1/procs
    ##$SF= 600.049944842336
    ```

    Procs SF = base spectrometer

    ```
    >>> (SFO1*1e6-SF*1e6)/600
    4.79013500670592
    ```

    Meant to be: 55.16
    """
    )
    return


@app.cell
def _():
    import re
    import numpy as np
    import xml.etree.ElementTree as ET

    data_dir = 'spectra' # Base directory of the 
    experiment_dir = '20250811_cit_nacit_titr' # Base directory of the experiment
    experiment_count = 24 # Number of experiments ran, 1 to x
    experiment_number = 3 # Which directory in the experiment contains the information

    experiments = []
    for i in range(1, experiment_count+1):
        experiments.append(f'{experiment_dir}_{i}')

    sfo1_values = []
    sf_values = []
    o1_values = []
    peak_values = []

    for experiment in experiments:
        with open(f'{data_dir}/{experiment}/{experiment_number}/acqus') as f:
            for line in f:
                match1 = re.search(r'\$SFO1=\s*([\d.]+)', line)
                match2 = re.search(r'\$O1=\s+([\d.]+)', line)
                if match1:
                    sfo1_values.append(float(match1.group(1)))
                if match2:
                    o1_values.append(float(match2.group(1)))

        with open(f'{data_dir}/{experiment}/{experiment_number}/pdata/1/procs') as f:
            for line in f:
                match = re.search(r'\$SF=\s*([\d.]+)', line)
                if match:
                    sf_values.append(float(match.group(1)))

        with open(f'{data_dir}/{experiment}/{experiment_number}/pdata/1/peaklist.xml') as f:
            xml_data = f.read()
            root = ET.fromstring(xml_data)

            peaks = []

    # Iterate through each Peak1D element and extract F1 and intensity
            for peak in root.findall('.//Peak1D'):
                f1 = float(peak.get('F1'))
                intensity = float(peak.get('intensity'))
                peaks.append([f1, intensity])


            peak_values.append(peaks)


    # Convert lists to numpy arrays with float64 type for higher precision
    sfo1_array = np.array(sfo1_values, dtype=np.float64)
    sf_array = np.array(sf_values, dtype=np.float64)
    o1_array = np.array(o1_values, dtype=np.float64)

    sr_values = []
    for o1, sf, sfo1 in zip(o1_array, sf_array, sfo1_array):
        print((sf*1e6-sfo1*1e6))
        print((o1+sf*1e6-sfo1*1e6))

        sr_values.append(o1+sf*1e6-sfo1*1e6)

    sr_array = np.array(sr_values, dtype=np.float64)
    ppm_shift = [x/600.5 for x in sr_array]

    for idx, peaks in enumerate(peak_values):
        for peak in peaks:
            peak[0] += ppm_shift[idx]

    # print(ppm_shift)
    # print(peak_values[0])

    return (peak_values,)


@app.cell
def _(peak_values):
    import matplotlib.pyplot as plt

    def _():
        avg_ppm = []
        for peaks in peak_values:
            average = 0
            for peak in peaks:
                average += peak[0]

            average /= len(peaks)

            avg_ppm.append(average)
        return avg_ppm

    avg_ppm = _()

    plt.plot(avg_ppm)

    return avg_ppm, plt


@app.cell
def _(avg_ppm, plt):
    import pandas as pd

    out_dir = 'experimental'

    imported = pd.read_csv(f'{out_dir}/eppendorfs.csv')

    phs = [x for x in imported['ph']]
    def _():
        dels = [5, 6, 16]
        for i in dels:
            del phs[i]
            del avg_ppm[i]

    _()

    # phs.sort()

    plt.figure(figsize=(14, 8))
    plt.plot(phs, avg_ppm)
    plt.xlabel('pH')
    plt.ylabel('Chemical shift (PPM)')
    plt.title('Chemical shift of Citrate Peaks Compared with pH Through Trisodium Citrate Speciation')
    plt.savefig('figs/ppmtoph.svg')

    figure = plt.gca()
    return (figure,)


@app.cell(hide_code=True)
def _(figure, mo):
    mo.md(rf"""{mo.as_html(figure)}""")
    return


if __name__ == "__main__":
    app.run()
