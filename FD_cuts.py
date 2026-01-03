#Created on January 3 2026 01:14
#Author: Kevin Arias

# importing libraries

import uproot
import numpy as np
import matplotlib.pyplot as plt
import vector as vec
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy.integrate import quad

# constants
mass_p = 0.938272088  # GeV/c^2 for proton
mass_pim = 0.13957     # GeV/c^2 for pion
mass_pip = 0.13957     # GeV/c^2 for pion
mass_k = 0.49367     # GeV/c^2 for kaon
mass_e = 0.000511     # GeV/c^2 for electron
E_beam = 10.2         # GeV (assuming beam energy of 10.6 GeV)
mass_n = .939565      # GeV/c^2 for neutron

# no idea who named these branches/trees

# Open the ROOT file and load the tree/data table which can hold different types of data
file = uproot.open('Pppim_eFD_all.root')
tree = file['Individual'] # I did not name this tree, I assumed it means data from individual events not binned or summed

# making arrays of momentum information for electron, pi minus, and both protons from called tree
# this is the scattered electron not the e_beam
px_e = tree['px_e'].array()
py_e = tree['py_e'].array()
pz_e = tree['pz_e'].array()

px_p1 = tree['px_p1'].array()
py_p1 = tree['py_p1'].array()
pz_p1 = tree['pz_p1'].array()

px_p2 = tree['px_p2'].array()
py_p2 = tree['py_p2'].array()
pz_p2 = tree['pz_p2'].array()

px_pim = tree['px_pim'].array()
py_pim = tree['py_pim'].array()
pz_pim = tree['pz_pim'].array()

def p_mag(x, y , z):
    return np.sqrt(x**2 + y**2 + z**2)
# magnitude calculations for later (GeV/c)
e_mag = p_mag(px_e, py_e, pz_e) 
p1_mag = p_mag(px_p1, py_p1, pz_p1)
p2_mag = p_mag(px_p2, py_p2, pz_p2)
pim_mag = p_mag(px_pim, py_pim, pz_pim)

# 32,911,203 data points each??

# extracting delta time information 
dt_e = tree['deltaTime_e'].array()
dt_p1 = tree['deltaTime_p1'].array()
dt_p2 = tree['deltaTime_p2'].array()
dt_pim = tree['deltaTime_pim'].array()

# extracting chi2pid information
e_chi2pid = tree['chi2pid_e'].array()
p1_chi2pid = tree['chi2pid_p1'].array()
p2_chi2pid = tree['chi2pid_p2'].array()
pim_chi2pid = tree['chi2pid_pim'].array()


# creating four vectors of inital conditions (beam and target)
# how do I know if my beam is well defined?
# not neglecting mass of electron
p_e_beam = np.array([E_beam, 0, 0, np.sqrt(E_beam**2 - mass_e**2)])
p_target = np.array([mass_p, 0, 0, 0])


# creating four vectors for final conditions (p1, p2, e scattered, pim)
p_e_scatter = np.column_stack([np.sqrt(mass_e**2 + e_mag**2), px_e, py_e, pz_e])
p_p1 = np.column_stack([np.sqrt(mass_p**2 + p1_mag**2), px_p1, py_p1, pz_p1])
p_p2 = np.column_stack([np.sqrt(mass_p**2 + p2_mag**2), px_p2, py_p2, pz_p2])
p_pim = np.column_stack([np.sqrt(mass_pim**2 + pim_mag**2), px_pim, py_pim, pz_pim])


#%% missing mass method

"""Cut on MM to only keep events where the 
missing particle looks like an neutron"""

N = len(px_e)

# P_missing = P_initial - sum of P_observed
p_initial_event = p_e_beam + p_target         #this has shape (4,)
p_initial = np.tile(p_initial_event, (N, 1))  #this has shape (N, 4)
p_observed = p_e_scatter + p_p1 + p_p2 + p_pim
p_mm = p_initial - p_observed

E_mm  = p_mm[:, 0]
px_mm = p_mm[:, 1]
py_mm = p_mm[:, 2]
pz_mm = p_mm[:, 3]

MM2 = E_mm**2 - (px_mm**2 + py_mm**2 + pz_mm**2)
MM2 = np.where(MM2 > 0, MM2, 0) # preventing small negatives
MM = np.sqrt(MM2)
plt.figure()
plt.hist(MM, bins=200, range=(0.01, 2.5), histtype='step', color='k')
plt.axvline(mass_n, linestyle='--', label=f'Antineutron {mass_n}GeV' )
plt.xlabel('Missing Mass (GeV)')
plt.ylabel('Counts')
plt.title('Missing Mass (no cuts)')
plt.tight_layout()
plt.savefig('MM_no_cuts.pdf')
plt.show()


#%% invariant mass of final hadronic system (W)
#(when you don't see/detect all final state particles)

""""Cut on W to only keep events where the whole hadronic system 
is in the kinematic window I want"""

p_W = p_initial - p_e_scatter

E_W  = p_W[:, 0]
px_W = p_W[:, 1]
py_W = p_W[:, 2]
pz_W = p_W[:, 3]


W2 = E_W**2 - (px_W**2 + py_W**2 + pz_W**2)
W = np.sqrt(W2)

plt.figure()
plt.hist(W, bins=100, range=(0.01, 2.5), histtype='step', color='k')
plt.xlabel('Hadronic Invariant Mass (no cuts)')
plt.ylabel('Counts')
plt.title('Hadronic Invariant Mass (no cuts)')
plt.tight_layout()
plt.savefig('W_no_cuts_ZOOM_OUT.pdf')
plt.show()

plt.figure()
plt.hist(W, bins=80, range=(0.7, 1.2), histtype='step', color='k')
plt.xlabel('Hadronic Invariant Mass (no cuts) (GeV)')
plt.ylabel('Counts')
plt.title('Hadronic Invariant Mass (no cuts)')
plt.tight_layout()
plt.savefig('W_no_cuts.pdf')
plt.show()







