# Settings for cell types, model types, and cell numbers

params = {}

# Cell types to include
params["cell_types"] = [
    "L23",
    "L5",
    "L6",
    "I23",
    "I5",
    "I6",
    "L23input",
    "L5input",
    "L6input",
    "I23input",
    "I5input",
    "I6input",
]

# Model type of cells
params["model_types"] = {
    "L23": "esser",
    "L5": "esser",
    "L6": "esser",
    "I23": "esser",
    "I5": "esser",
    "I6": "esser",
    "L23input": "poisson",
    "L5input": "poisson",
    "L6input": "poisson",
    "I23input": "poisson",
    "I5input": "poisson",
    "I6input": "poisson",
}

# Firing rate (Hz) for poisson model type
params["firing_rate"] = {
    "L23": 0,
    "L5": 0,
    "L6": 0,
    "I23": 0,
    "I5": 0,
    "I6": 0,
    "L23input": 0.25,
    "L5input": 0.25,
    "L6input": 0.25,
    "I23input": 0.25,
    "I5input": 0.25,
    "I6input": 0.25,
}

# Cell numbers
params["N"] = {
    "L23": 158,
    "L5": 158,
    "L6": 158,
    "I23": 79,
    "I5": 79,
    "I6": 79,
    "L23input": 79,
    "L5input": 79,
    "L6input": 79,
    "I23input": 79,
    "I5input": 79,
    "I6input": 79,
}
