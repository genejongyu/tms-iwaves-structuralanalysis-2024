# Defining color-blind friendly color palettes

import tol_colors as tc

cmap_iridescent = tc.tol_cmap("iridescent")

cset_bright = list(tc.tol_cset("bright"))
cset_light = list(tc.tol_cset("light"))
cset_vibrant = list(tc.tol_cset("vibrant"))
cset_high_contrast = list(tc.tol_cset("high-contrast"))
cset_muted = list(tc.tol_cset("muted"))

colors = {}
colors["waves"] = {}
colors["waves"]["D"] = cset_vibrant[1]
colors["waves"]["I1"] = cset_vibrant[5]
colors["waves"]["I2"] = cset_vibrant[0]
colors["waves"]["I3"] = cset_vibrant[3]

colors["spikes"] = {}
colors["spikes"]["L2/3 PC"] = cset_vibrant[4]
colors["spikes"]["L2/3 INT"] = cset_vibrant[2]
colors["spikes"]["L5 PC"] = cset_vibrant[0]
colors["spikes"]["L5 INT"] = cset_vibrant[5]
colors["spikes"]["L6 PC"] = cset_vibrant[1]
colors["spikes"]["L6 INT"] = cset_vibrant[3]

colors["spikes"]["L23"] = cset_vibrant[4]
colors["spikes"]["I23"] = cset_vibrant[2]
colors["spikes"]["L5"] = cset_vibrant[0]
colors["spikes"]["I5"] = cset_vibrant[5]
colors["spikes"]["L6"] = cset_vibrant[1]
colors["spikes"]["I6"] = cset_vibrant[3]

colors["spikes"]["L23input"] = cset_vibrant[7]
colors["spikes"]["I23input"] = cset_vibrant[7]
colors["spikes"]["L5input"] = cset_vibrant[7]
colors["spikes"]["I5input"] = cset_vibrant[7]
colors["spikes"]["L6input"] = cset_vibrant[7]
colors["spikes"]["I6input"] = cset_vibrant[7]

colors["popsmooth"] = {}
colors["popsmooth"]["D+"] = cset_bright[0]
colors["popsmooth"]["D-"] = cset_bright[1]
colors["popsmooth"]["2013_31"] = cset_bright[0]
colors["popsmooth"]["2020_43"] = cset_bright[1]