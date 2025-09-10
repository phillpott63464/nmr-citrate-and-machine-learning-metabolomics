# Navigation

- ROCM enabled docker image for AMD gpus:
  - docker-compose.yml
  - Dockerfile
- ML Model Notes.md: File of model parameters, testing metrics, and evaluations
- chelationnotes.md: Notes on solbuility of different citrate metal complexes

## App

- CUDA enabled UV setup:
  - pyproject.toml
  - uv.lock
- CONDA environment for docker image:
  - environment.yml
  - requirements.txt
- bufferdata.csv: Table of citrate/citric acid buffer pHs and ratios taken from Data for Biochemical Research, Third Edition (Make this a full reference)
- PKA_STUDY.db: Optuna database of trials used to determine pKas from bufferdata.
- metal_experiments.csv - Spreadsheet containing concentrations of components used to create samples for chelation experiments, including stock requirements.
- new_metal_experiments.csv - Spreadsheet containing concentrations of components not yet used.

Marimo files, all containing self-documentation, run with uv run marimo edit from app and follow provided link or docker compose up -d from root directory and go to url localhost:27182:
- Experiment Definition.py: calculates speciation and pkas, defines experiment parameters
- Experiments load: Loads experiment results, performs fitting and machine learning on results
- Final Single Metabolite: Uses machine learning and Morgan's spectra generation code to determine classification and concentration (as ratio of reference to citrate) of citrate in complex mixtures.
- Randomisation Hold Back: Uses machine learning and Morgan's spectra generation code to determine classification and concentration of unseen (not in training data) randomised metabolites in complex mixture.
- Test Morgan.py: Tests morgan's  using cProfile and pStats. Used during optimisation of qm. 
- View Morgans Spectra.py: Views spectra from morganspectra. 

### experimental

Folder containing CSV files of experimental samples, including blanks. Includes metal_eppendorfs for chelation and na-eppendorfs for speciation.

### figs

SVGs of all figures generated in python files.

### manim

Experimental animated figures using [manim community](https://github.com/manimCommunity/manim) ([MIT License](licenses/LICENSE-3b1b)) ([MIT License](licenses/LICENSE-manimcommunity))

### morgancode

Folder containing all the spectra generation code provided by Morgan, as well as modified qm system from [nmrsim](https://github.com/sametz/nmrsim) ([MIT license](licenses/LICENSE-nmrsim))

### morganspectra

Folder containing all spectra of synthetic mixtures provided by Morgan.

- Synthetic_Mixture_Lookup.xlsx - spreadsheet of samples

#### 2023-02-16_Synthetic_Mixtures

Folder of bruker format data

### phfork

Folder containing code from [phfork](https://github.com/mhvwerts/pHfork) (No license)

### spectra

Folder containing 1H bruker data from the experimental samples.

#### 20250811_cit_nacit_titr_x, x=1-24

#### 20250811_cit_ca_mg_cit_titr_x, x=1-24

#### 20250811_cit_ca_mg_cit_titr_17_rep

Due to errors with bubbles in the sample

#### 20250811_cit_ca_mg_cit_titr_22_rep

Due to errors with bubbles in the sample
