Code associated with Yu GJ, Ranieri F, Di Lazzaro V, Sommer MA, Peterchev AV, Grill WM (2024) 
"Circuits and mechanisms for TMS-induced corticospinal waves: Connecting sensitivity analysis to the network graph"

1. Install python dependencies (recommended to install into a virtual environment).
2. Full data generated during optimization and TVAT analysis can be downloaded from https://doi.org/10.5281/zenodo.10729433
    - Save to data folder.

The analysis folder contains classes, libraries, and scripts used to generate the files in the data folder.
- Results are saved to data folder by default.

The data folder contains the data used in the analysis and generate_figures scripts.

The generate_figures folder contains scripts to generate plots and subplots of main figures and supplemental figures.

The model folder contains the classes and scripts that require instantiating the network model.
- This includes classes and scripts for launching particle swarm optimization and TVAT analysis.
- Compile mechanisms by moving the the model directory and running nrnivmodl mod
- Results are saved to data folder by default.

Dependencies:
- h5py 3.10.0
- kneed 0.8.5
- matplotlib 3.7.2
- mpi4py 3.1.5
- multiprocess 0.70.15
- neuron 8.2.3
- networkx 3.2.1
- numpy 1.26.4
- openmpi 4.1.6
- python 3.10.13
- scikit-learn 1.4.0
- scipy 1.12.0
- statsmodels 0.14.1