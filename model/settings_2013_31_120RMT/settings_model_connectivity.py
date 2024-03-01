# Settings for connectivity
# p - probability of connection (divergence)
# strength - base synaptic weight scaling factor
# delay mean - mean delay for AP conduction
# delay std - standard deviation for AP conduction delay
# syn_types - synapse types for connection

params = {}

params["conduction_velocity"] = 570 # units: microns/ms

params["diameter_column"] = 500 # units: microns
params["spacing_column"] = 50 # units: microns

params["depth"] = {} # units: microns
params["depth"]["L1"] = 300
params["depth"]["L23"] = 1350
params["depth"]["I23"] = 1350
params["depth"]["L5"] = 2250
params["depth"]["I5"] = 2250
params["depth"]["L6"] = 3000
params["depth"]["I6"] = 3000

params["N_per_column"] = {}
params["N_per_column"]["L23"] = 2
params["N_per_column"]["I23"] = 1
params["N_per_column"]["L5"] = 2
params["N_per_column"]["I5"] = 1
params["N_per_column"]["L6"] = 2
params["N_per_column"]["I6"] = 1

params["con"] = {}
params["con"]["p"] = {}
params["con"]["p"]["L23"] = {
    "L23": 0.05,
    "L5": 1,
    "L6": 1,
    "I23": 0.05,
    "I5": 1,
    "I6": 1,
}

params["con"]["p"]["L5"] = {
    "L23": 1,
    "L5": 0.025,
    "L6": 1,
    "I23": 1,
    "I5": 0.025,
    "I6": 1,
}

params["con"]["p"]["L6"] = {
    "L5": 1, 
    "L6": 0.025, 
    "I5": 1, 
    "I6": 0.025
}

params["con"]["p"]["I23"] = {
    "L23": 0.2, 
    "L5": 0.25, 
    "L6": 0.25, 
    "I23": 0.2
}

params["con"]["p"]["I5"] = {
    "L5": 0.2, 
    "I5": 0.2
}

params["con"]["p"]["I6"] = {
    "L6": 0.2, 
    "I6": 0.2
}

params["con"]["p"]["L23input"] = {"L23": 0.125}
params["con"]["p"]["L5input"] = {"L5": 0.125}
params["con"]["p"]["L6input"] = {"L6": 0.125}
params["con"]["p"]["I23input"] = {"I23": 0.125}
params["con"]["p"]["I5input"] = {"I5": 0.125}
params["con"]["p"]["I6input"] = {"I6": 0.125}

params["con"]["strength"] = {}
params["con"]["strength"]["L23"] = {
    "L23": {"AMPA": 2, "NMDA": 2},
    "L5": {"AMPA": 2, "NMDA": 2},
    "L6": {"AMPA": 0.5, "NMDA": 0.5},
    "I23": {"AMPA": 2, "NMDA": 2},
    "I5": {"AMPA": 2, "NMDA": 2},
    "I6": {"AMPA": 0.5, "NMDA": 0.5},
}

params["con"]["strength"]["L5"] = {
    "L23": {"AMPA": 1, "NMDA": 1},
    "L5": {"AMPA": 2, "NMDA": 2},
    "L6": {"AMPA": 1, "NMDA": 1},
    "I23": {"AMPA": 1, "NMDA": 1},
    "I5": {"AMPA": 2, "NMDA": 2},
    "I6": {"AMPA": 1, "NMDA": 1},
}

params["con"]["strength"]["L6"] = {
    "L5": {"AMPA": 0.25, "NMDA": 0.25},
    "L6": {"AMPA": 2, "NMDA": 2},
    "I5": {"AMPA": 0.25, "NMDA": 0.25},
    "I6": {"AMPA": 2, "NMDA": 2},
}

params["con"]["strength"]["I23"] = {
    "L23": {"GABAA": 2, "GABAB": 1},
    "L5": {"GABAA": 1, "GABAB": 0.5},
    "L6": {"GABAA": 1, "GABAB": 0.5},
    "I23": {"GABAA": 2, "GABAB": 1},
}

params["con"]["strength"]["I5"] = {
    "L5": {"GABAA": 1, "GABAB": 1},
    "I5": {"GABAA": 1, "GABAB": 1},
}

params["con"]["strength"]["I6"] = {
    "L6": {"GABAA": 1, "GABAB": 1},
    "I6": {"GABAA": 1, "GABAB": 1},
}

params["con"]["strength"]["L23input"] = {"L23": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["L5input"] = {"L5": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["L6input"] = {"L6": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["I23input"] = {"I23": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["I5input"] = {"I5": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["I6input"] = {"I6": {"AMPA": 1, "NMDA": 1}}

params["con"]["sigma"] = {}
params["con"]["sigma"]["L23"] = {
    "L23": 12 * 50 / 3,
    "L5": 2 * 50 / 3,
    "L6": 2 * 50 / 3,
    "I23": 12 * 50 / 3,
    "I5": 2 * 50 / 3,
    "I6": 2 * 50 / 3,
}

params["con"]["sigma"]["L5"] = {
    "L23": 2 * 50 / 3,
    "L5": 12 * 50 / 3,
    "L6": 2 * 50 / 3,
    "I23": 2 * 50 / 3,
    "I5": 12 * 50 / 3,
    "I6": 2 * 50 / 3,
}

params["con"]["sigma"]["L6"] = {
    "L5": 2 * 50 / 3, 
    "L6": 9 * 50 / 3, 
    "I5": 2 * 50 / 3, 
    "I6": 9 * 50 / 3
}

params["con"]["sigma"]["I23"] = {
    "L23": 7 * 50 / 3, 
    "L5": 2 * 50 / 3, 
    "L6": 2 * 50 / 3, 
    "I23": 7 * 50 / 3
}

params["con"]["sigma"]["I5"] = {
    "L5": 7 * 50 / 3, 
    "I5": 7 * 50 / 3
}

params["con"]["sigma"]["I6"] = {
    "L6": 7 * 50 / 3, 
    "I6": 7 * 50 / 3
}

params["con"]["sigma"]["L23input"] = {"L23": 8 * 50 / 3}
params["con"]["sigma"]["L5input"] = {"L5": 8 * 50 / 3}
params["con"]["sigma"]["L6input"] = {"L6": 8 * 50 / 3}
params["con"]["sigma"]["I23input"] = {"I23": 8 * 50 / 3}
params["con"]["sigma"]["I5input"] = {"I5": 8 * 50 / 3}
params["con"]["sigma"]["I6input"] = {"I6": 8 * 50 / 3}

params["con"]["delay_mean"] = {}
params["con"]["delay_mean"]["L23"] = {
    "L23": 1.0,
    "L5": 2.0,
    "L6": 3.0,
    "I23": 1.0,
    "I5": 2.0,
    "I6": 3.0,
}

params["con"]["delay_mean"]["L5"] = {
    "L23": 2.0,
    "L5": 1.0,
    "L6": 2.0,
    "I23": 2.0,
    "I5": 1.0,
    "I6": 2.0,
}

params["con"]["delay_mean"]["L6"] = {"L5": 2.0, "L6": 1.0, "I5": 2.0, "I6": 1.0}

params["con"]["delay_mean"]["I23"] = {"L23": 1.0, "L5": 2.0, "L6": 3.0, "I23": 1.0}

params["con"]["delay_mean"]["I5"] = {"L5": 1.0, "I5": 1.0}

params["con"]["delay_mean"]["I6"] = {"L6": 1.0, "I6": 1.0}

params["con"]["delay_mean"]["L23input"] = {"L23": 1.67}
params["con"]["delay_mean"]["L5input"] = {"L5": 1.67}
params["con"]["delay_mean"]["L6input"] = {"L6": 1.67}
params["con"]["delay_mean"]["I23input"] = {"I23": 1.67}
params["con"]["delay_mean"]["I5input"] = {"I5": 1.67}
params["con"]["delay_mean"]["I6input"] = {"I6": 1.67}

params["con"]["delay_std"] = {}
params["con"]["delay_std"]["L23"] = {
    "L23": 0.1,
    "L5": 0.61,
    "L6": 1.68,
    "I23": 0.1,
    "I5": 0.61,
    "I6": 1.68,
}

params["con"]["delay_std"]["L5"] = {
    "L23": 0.61,
    "L5": 0.1,
    "L6": 1.57,
    "I23": 0.61,
    "I5": 0.1,
    "I6": 1.57,
}

params["con"]["delay_std"]["L6"] = {"L5": 1.57, "L6": 0.1, "I5": 1.57, "I6": 0.1}

params["con"]["delay_std"]["I23"] = {"L23": 0.1, "L5": 0.61, "L6": 1.68, "I23": 0.1}

params["con"]["delay_std"]["I5"] = {"L5": 0.1, "I5": 0.1}

params["con"]["delay_std"]["I6"] = {"L6": 0.1, "I6": 0.1}

params["con"]["delay_std"]["L23input"] = {"L23": 0.4}
params["con"]["delay_std"]["L5input"] = {"L5": 0.4}
params["con"]["delay_std"]["L6input"] = {"L6": 0.4}
params["con"]["delay_std"]["I23input"] = {"I23": 0.4}
params["con"]["delay_std"]["I5input"] = {"I5": 0.4}
params["con"]["delay_std"]["I6input"] = {"I6": 0.4}

params["con"]["syn_types"] = {}
params["con"]["syn_types"]["L23"] = {
    "L23": ["AMPA", "NMDA"],
    "L5": ["AMPA", "NMDA"],
    "L6": ["AMPA", "NMDA"],
    "I23": ["AMPA", "NMDA"],
    "I5": ["AMPA", "NMDA"],
    "I6": ["AMPA", "NMDA"],
}

params["con"]["syn_types"]["L5"] = {
    "L23": ["AMPA", "NMDA"],
    "L5": ["AMPA", "NMDA"],
    "L6": ["AMPA", "NMDA"],
    "I23": ["AMPA", "NMDA"],
    "I5": ["AMPA", "NMDA"],
    "I6": ["AMPA", "NMDA"],
}

params["con"]["syn_types"]["L6"] = {
    "L5": ["AMPA", "NMDA"],
    "L6": ["AMPA", "NMDA"],
    "I5": ["AMPA", "NMDA"],
    "I6": ["AMPA", "NMDA"],
}

params["con"]["syn_types"]["I23"] = {
    "L23": ["GABAA", "GABAB"],
    "L5": ["GABAA", "GABAB"],
    "L6": ["GABAA", "GABAB"],
    "I23": ["GABAA", "GABAB"],
}

params["con"]["syn_types"]["I5"] = {"L5": ["GABAA", "GABAB"], "I5": ["GABAA", "GABAB"]}

params["con"]["syn_types"]["I6"] = {"L6": ["GABAA", "GABAB"], "I6": ["GABAA", "GABAB"]}

params["con"]["syn_types"]["L23input"] = {"L23": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["L5input"] = {"L5": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["L6input"] = {"L6": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["I23input"] = {"I23": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["I5input"] = {"I5": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["I6input"] = {"I6": ["AMPA", "NMDA"]}
