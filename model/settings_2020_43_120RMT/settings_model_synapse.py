# Settings for base synapse parameters

params = {}

params["syn"] = {}

params["syn"]["AMPA"] = {"tau1": 0.5, "tau2": 2.4, "gpeak": 0.1, "e": 0}

params["syn"]["NMDA"] = {"tau1": 4, "tau2": 40, "gpeak": 0.1, "e": 0}

params["syn"]["GABAA"] = {"tau1": 1, "tau2": 7, "gpeak": 0.33, "e": -70}

params["syn"]["GABAB"] = {"tau1": 60, "tau2": 200, "gpeak": 0.0132/4, "e": -90}
