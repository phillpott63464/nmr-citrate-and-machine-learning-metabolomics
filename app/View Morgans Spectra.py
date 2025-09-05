import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import os
    import re
    import numpy as np
    import nmrglue as ng
    return mo, ng, np, os, re


@app.cell
def _(ng, np, os, re):
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

    def bruker_fft(data_dir, experiment, experiment_number):
        """
        Convert time domain data to frequency domain
        https://github.com/jjhelmus/nmrglue/blob/master/examples/bruker_processed_1d/bruker_processed_1d.py
        """


        phc = extract_phc(
            data_dir=data_dir,
            experiment_number=experiment_number,
            experiment=experiment,
        )

        data, dic = read_bruker(data_dir, experiment, experiment_number)
        # data = read_fid(data_dir=data_dir, experiment=experiment, experiment_number=experiment_number)

        # Process the spectrum
        # data = ng.proc_base.zf_size(
        #     data, 2**20
        # )    # Zero fill to 32768 points
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
    return (bruker_fft,)


@app.cell
def _():
    data_dir = 'Morgan-Spectra/2023-02-16_Synthetic_Mixtures'   # The directory all data is in

    import pandas as pd

    file = pd.ExcelFile('Morgan-Spectra/Synthetic_Mixture_Lookup.xlsx')

    spectra_sheet = pd.read_excel(file,'Spectra')

    indices = [x for x, y in zip(spectra_sheet['spectrumFileName'], spectra_sheet['experimentType']) if y == 'H']
    return data_dir, indices


@app.cell
def _():
    return


@app.cell
def _(bruker_fft, data_dir, indices, np):
    from matplotlib import pyplot as plt

    data = {}
    for experiment in indices:
        data[experiment] = np.array(bruker_fft(data_dir, experiment, experiment_number=''))

        if max(data[experiment][1]) < min(data[experiment][1]):
            data[experiment][1] = data[experiment][1] * -1

        lower_bound, upper_bound = 2.4, 2.8

        x = data[experiment][0]

        mask = (x >= lower_bound) & (x <= upper_bound)
    
        data[experiment] = data[experiment][:, mask]
    return data, plt


@app.cell
def _(plt):
    def gen_fig(data, peaks=None):
        figure = plt.figure()
        plt.plot(*data)

        if peaks is not None:
            scatterx, scattery = peaks[:, 0], peaks[:, 1]
            plt.scatter(scatterx, scattery, color='RED')
        return figure
    return (gen_fig,)


@app.cell
def _(data, gen_fig, ng, np):
    figs = {}
    peaks = {}

    for set_name in data:
        arr = data[set_name]          # shape (2, 32000)
        y = arr[1]
        threshold = np.percentile(y, 99)

        ymask = (y >= threshold)

        # peaks[set_name] = data[set_name][:, ymask]

        # print(len(peaks[set_name][0]))

        peak_positions = [x[1] for x in ng.analysis.peakpick.pick(data[set_name], threshold)]
        peak_positions = [data[set_name][:, int(i)] for i in peak_positions]
        peak_positions = np.vstack(peak_positions)
        peaks[set_name] = peak_positions
    
        figs[set_name] = gen_fig(data[set_name], peaks[set_name])

        # if len(figs) > 0:
        #     die

    # Now you can show any figure later using figs[index]
    return (figs,)


@app.cell
def _(figs, mo):
    mo.mpl.interactive(figs[34])
    return


if __name__ == "__main__":
    app.run()
