# This is a script to plot the energy spectrum of antineutron candidates from the ECAL

%matplotlib qt 
import uproot # to read ROOT files
import awkward as ak # to handle jagged arrays from ROOT
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for plotting 
import vector as vec

# constants
mass_p = 0.938272088  # GeV/c^2 for proton
mass_pim = 0.13957     # GeV/c^2 for pion
mass_pip = 0.13957     # GeV/c^2 for pion
mass_k = 0.49367     # GeV/c^2 for kaon
mass_e = 0.000511     # GeV/c^2 for electron
E_beam = 10.1998      # GeV (assuming beam energy of 10.6 GeV)
mass_n = .939565      # GeV/c^2 for neutron

# openning root file and accessing TTree
file = uproot.open("v6_kev_Pppim_eFT_all.root")
tree = file["Individual"]

# loading kinematic variables for known tracks (electron, 2 protons, pi-minus)
px_e, py_e, pz_e = tree["px_e"].array(), tree["py_e"].array(), tree["pz_e"].array()
px_p1, py_p1, pz_p1 = tree["px_p1"].array(), tree["py_p1"].array(), tree["pz_p1"].array()
px_p2, py_p2, pz_p2 = tree["px_p2"].array(), tree["py_p2"].array(), tree["pz_p2"].array()
px_pim, py_pim, pz_pim = tree["px_pim"].array(), tree["py_pim"].array(), tree["pz_pim"].array()

# loading calorimeter jagged arrays for orphaned ("non tracked") hits
orphan_E = tree["orphan_E"].array()

# loading components of orphaned hits to manually compute angles later
orphan_x, orphan_y, orphan_z = tree["orphan_x"].array(), tree["orphan_y"].array(), tree["orphan_z"].array()

# plotting raw orphaned hit energy spectrum
plt.figure()
plt.hist(ak.flatten(orphan_E), bins=100, range=(0, 3), histtype='step', color='red', alpha=1)
plt.title("Raw Energy Spectrum of Orphaned Calorimeter Hits", fontsize=16)
plt.xlabel("Energy from ECAL(GeV)", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.yscale('log') # log scale to see low energy hits better
plt.grid(True)
plt.show()
plt.savefig("raw_orphan_energy_spectrum.pdf")

orphan_list = ak.flatten(orphan_E)
orphans = ak.sort(orphan_list, ascending=False) # sort energies in descending order for each event

print(f"Total orphan hits: {len(orphan_E)}")
print("Highest 10 Orphan Hit Energies (GeV):")
for i, energies in enumerate(orphans[:10], 1):
    print(f"{i}: {energies:.3f} GeV")

# missing mass calculation 
p_beam = vec.obj(px = 0, py = 0, pz = 10.2, E = 10.2)
p_target = vec.obj(px = 0, py = 0, pz = 0, M = 0.938272088)

p_e = vec.array({'px': px_e, "py": py_e, "pz": pz_e, "M": np.ones_like(px_e) * mass_e})
p_p1 = vec.array({'px': px_p1, "py": py_p1, "pz": pz_p1, "M": np.ones_like(px_p1) * mass_p})
p_p2 = vec.array({'px': px_p2, "py": py_p2, "pz": pz_p2, "M": np.ones_like(px_p2) * mass_p})
p_pim = vec.array({'px': px_pim, "py": py_pim, "pz": pz_pim, "M": np.ones_like(px_pim) * mass_pim})

p_miss = p_beam + p_target - p_e - p_p1 - p_p2 - p_pim

# .px, .py, .pz give the components of the missing momentum vector
# calculate momentum magnitudes
r_hit = np.sqrt(orphan_x**2 + orphan_y**2 + orphan_z**2)
mag_p_miss = np.sqrt(p_miss.px**2 + p_miss.py**2 + p_miss.pz**2)

# calculate angles between missing momentum vector and orphaned hits
dot_product = (orphan_x * p_miss.px) + (orphan_y * p_miss.py) + (orphan_z * p_miss.pz)
cos_theta = dot_product / (r_hit * mag_p_miss)
# clamp to [-1, 1] to avoid numerical issues
cos_theta = ak.where(cos_theta > 1.0, 1.0, cos_theta)
cos_theta = ak.where(cos_theta < -1.0, -1.0, cos_theta)
theta = np.arccos(cos_theta)

# best theta for each event (smallest angle between any orphan hit and missing momentum vector)
best_theta = ak.argmin(theta, axis=1, keepdims=True) 

# phi angle in the transverse plane
phi = np.arctan2(orphan_y, orphan_x) - np.arctan2(p_miss.py, p_miss.px)
best_phi = ak.argmin(np.abs(phi), axis=1, keepdims=True)

# best energy based on best theta
best_energy = ak.flatten(orphan_E[best_theta])

plt.figure()
plt.hist(best_energy, bins=100, range=(0, 3), histtype='step', color='blue', alpha=1)
plt.title("Energy of Best Candidate Hit", fontsize=16)
plt.axvline(x=2*mass_n, color='red', linestyle='--', linewidth=2, label=r'${\bar{n}}_{mass} + n_{mass}$')
plt.xlabel("Energy from ECAL(GeV)", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
plt.savefig("best_candidate_energy_spectrum.pdf")

candidates = ak.sort(best_energy, ascending=False) # sort energies in descending order for each event

print(f"Total best candidate hits: {len(best_energy)}")
print(f"Highest 10 Best Candidate Energies (GeV):")
for i, energy in enumerate(candidates[:10], 1):
    print(f"{i}: {energy:.3f} GeV")
# %%
