#%% PERPLEXITY CODE
%matplotlib qt
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from upkit import Histo, Histo2D, RootAnalysis
from upkit.hists import *
import vector as vec

set_plot_style()
t = RootAnalysis("v2_kev_Pppim_eFT_006667.root", 'Individual')

branches = ["miss_mass", "Enbar_calo", 'px_p1', 'py_p1', 'pz_p1', 'px_p2', 'py_p2', 'pz_p2', 'px_e', 'py_e', 'pz_e', 'px_pim', 'py_pim', 'pz_pim', 'nonprim_x', 'nonprim_y', 'nonprim_z', 'nonprim_E']
t.load_branches(branches)

p_p1 = vec.array({"px": t.data["px_p1"], "py": t.data["py_p1"], "pz": t.data["pz_p1"], "M": np.ones_like(t.data["px_p1"]) * 0.938})
p_p2 = vec.array({"px": t.data["px_p2"], "py": t.data["py_p2"], "pz": t.data["pz_p2"], "M": np.ones_like(t.data["px_p2"]) * 0.938})
p_e  = vec.array({"px": t.data["px_e"], "py": t.data["py_e"], "pz": t.data["pz_e"], "M": np.ones_like(t.data["px_e"]) * 0.000511})
p_pim = vec.array({"px": t.data["px_pim"], "py": t.data["py_pim"], "pz": t.data["pz_pim"], "M": np.ones_like(t.data["px_pim"]) * 0.13957})

P_beam = vec.obj(px=0, py=0, pz=10.6, M=0.000511)
P_target = vec.obj(px=0, py=0, pz=0, M=0.938)
P_miss = P_beam + P_target - p_e - p_p1 - p_p2 - p_pim




#%%
mm    = t.data["miss_mass"]
Enbar = t.data["Enbar_calo"]
ang   = t.data["neutral_angle"]
# hasN  = t.data["has_neutral"]

mm    = ak.to_numpy(mm)
Enbar = ak.to_numpy(Enbar)
ang   = ak.to_numpy(ang)
# hasN  = ak.to_numpy(hasN)

# 1) Apply neutron-like MM window
mm_low, mm_high = 0.85, 1.05
mm_window = (mm > mm_low) & (mm < mm_high)

# 2) Keep events with a anti neutron candidate and valid angle
neutral_ok = (hasN == 1)
sel_angle = mm_window & neutral_ok

ang_sel = np.rad2deg(ang[sel_angle])

print(f"Total events: {t.num_entries}")
print("Events in MM window:", np.count_nonzero(mm_window))
print("Events with neutral + valid angle:", len(ang_sel))

#%%
# 3) Histogram of neutral_angle
plt.figure()
plt.hist(ang_sel, bins=60, range=(0.0, 180), histtype="step")
plt.xlabel("neutral_angle (deg)")
plt.ylabel("Counts")
plt.title("Angle between neutral ECAL hit and missing momentum")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
# 4) After inspecting this plot, pick a theta_max
theta_max = 30 # adjust after looking at the distribution
angle_cut = (np.rad2deg(ang) > 0.0) & (np.rad2deg(ang) < theta_max)

sel = mm_window & neutral_ok & angle_cut
Enbar_sel = Enbar[sel]

print("Events after angle cut:", np.count_nonzero(sel))

plt.figure()
plt.hist(Enbar_sel, bins=60, range=(0.0, 5.0), histtype="step")
plt.xlabel("Enbar_calo (GeV)")
plt.ylabel("Counts")
plt.title(f"Antineutron candidate energy (neutral_angle < {theta_max} deg)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print('---End of Perplexity Code----')
#%%
Histo2D(mm, np.rad2deg(ang), bins = 10, range = ((0.75, 1.25), (0, 35)), norm='log')
plt.axvline(0.939)

# %%
