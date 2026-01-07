#Created on January 3 2026 01:14
#Author: Kevin Arias

# importing libraries
#%%
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
#%%
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
#%%
p_beam = vec.obj(px = 0, py = 0, pz = 10.2, E = 10.2)
p_target = vec.obj(px = 0, py = 0, pz = 0, M = 0.938272088)

p_e = vec.array({'px': px_e, "py": py_e, "pz": pz_e, "M": np.ones_like(px_e) * mass_e})
p_p1 = vec.array({'px': px_p1, "py": py_p1, "pz": pz_p1, "M": np.ones_like(px_p1) * mass_p})
p_p2 = vec.array({'px': px_p2, "py": py_p2, "pz": pz_p2, "M": np.ones_like(px_p2) * mass_p})
p_pim = vec.array({'px': px_pim, "py": py_pim, "pz": pz_pim, "M": np.ones_like(px_pim) * mass_pim})


#%% missing mass method

N = len(px_e)


# MM2 = E_mm**2 - (px_mm**2 + py_mm**2 + pz_mm**2)
MM_vec = p_beam + p_target - p_e - p_p1 - p_p2 - p_pim
plt.figure()
plt.hist(np.array(MM_vec.M), bins=200, range=(0.01, 2.5), histtype='step', color='k')
plt.axvline(mass_n, linestyle='--', label=f'Antineutron {mass_n}GeV' )
plt.xlabel('Missing Mass(GeV)')
plt.ylabel('Counts')
plt.title('Missing Mass Distribution (no cuts)')

plt.text(
    1.5, 70000,
    'Foward Detector Data',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)

plt.tight_layout()
plt.savefig('MM_no_cuts.pdf')
plt.show()




#%% invariant mass of final hadronic system (W)

p_W = p_beam + p_target - p_e

plt.figure()
plt.hist(p_W.M, bins=100, range = (0, 5), histtype='step', color='k')
plt.axvline(2*mass_p + mass_pim + mass_n, color = 'red')
plt.xlabel('W (no cuts)')
plt.ylabel('Counts')
plt.title('Hadronic Invariant Mass Distribution(no cuts)')


plt.text(
    1.5, 0.95e6,
    r'$2m_p + m_{\pi} + m_n$',
    color='red',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig('W_no_cuts_ZOOM_OUT.pdf')
plt.show()





# %% This is a CUT!!! 
# Data is like a window and a cut is like a window cover

# cut_W = p_W.M > (2*mass_p + mass_pim + mass_n)
cut_W = (p_W.M > 3.1) & (p_W.M <= 4.09)

plt.figure()
plt.hist2d(np.array(p_W.M), np.array(MM_vec.M), bins = 100, range = ((0, 5), (0, 2.5)), norm = 'log')
plt.xlabel('W Distribution(GeV)')
plt.ylabel('Missing Mass Distribution(GeV)')
plt.title('MM vs W')
plt.text(
    3.2, 0.9,
    '                          ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)
plt.tight_layout()
plt.savefig('MM_vs_W.pdf')
plt.show()

# Now with cut
plt.figure()
plt.hist2d(np.array(p_W.M[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 5), (0, 2.5)), norm = 'log')
plt.xlabel('W (GeV)')
plt.ylabel('Missing Mass distribution(GeV)')
plt.title('MM vs W with threshold cut')
plt.text(
    0.7, 1.5,
    'Threshold cut: 3.1 < W < 4.09 GeV',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)
plt.tight_layout()
plt.savefig('MM_vs_W_threshold_cut.pdf')
plt.show()




# %% Missing Mass plot showing threshold cut
plt.figure()
plt.hist(MM_vec.M, bins = 20, range = (0.85, 1.15), histtype = 'step', color = 'black')
plt.hist(MM_vec.M[~cut_W], bins = 20, range = (0.85, 1.15), color = 'green', alpha = 0.5)
plt.hist(MM_vec.M[cut_W], bins = 20, range = (0.85, 1.15), color = 'blue', alpha = 0.5)
plt.xlabel('Missing Mass(GeV)')
plt.ylabel('Counts')
plt.title('Missing Mass distribution showing threshold cut')

y0=19000
plt.text(
    1.025, y0,
    'Total distribution',
    color='black',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)
plt.text(
    1.025, y0-2000,
    'Background',
    color='green',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)
plt.text(
    1.025, y0-4000,
    'Signal',
    color='blue',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)

plt.tight_layout()
plt.savefig('MM_threshold_cut.pdf')
plt.show()





# %% These are 2d histograms of MM vs W with a momemtum magnitude cut and threshold cut with reguards to different particles
plt.figure()
plt.hist2d(np.array(p_p1.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel('Proton 1 magnitude with threshold cut(GeV)')
plt.ylabel('MM with threshold cut')
plt.title('proton 1 momemtum magnitude vs W (with threshold cut)')
plt.savefig('p1_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_p2.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel('Proton 2 magnitude with threshold cut(GeV)')
plt.ylabel('MM with threshold cut')
plt.title('proton 2 momemtum magnitude vs W (with threshold cut)')
plt.savefig('p2_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_e.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r"$e\text{'}$ momentum magnitude with threshold cut(GeV)")
plt.ylabel('MM with threshold cut')
plt.title(r"$e\text{'}$ momemtum magnitude vs W (with threshold cut)")
plt.savefig('e_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_pim.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r"$\pi^-$ momentum magnitude with threshold cut(GeV)")
plt.ylabel('MM with threshold cut')
plt.title(r"$\pi^-$ momemtum magnitude vs W (with threshold cut)")
plt.savefig('pi_mom_mag_threshold.pdf')
plt.show()



# %% This is a momentum magnitude cut using all particles

cut_mag = cut_W & (p_p1.mag > 1.771) & (p_p1.mag < 4.831) & (p_pim.mag > 0.217) & (p_pim.mag < 2.232) & (p_e.mag > 0.911) & (p_e.mag < 3.363) & (p_p2.mag > 0.911) & (p_p2.mag < 2.730)
# to get these numbers, I went into spyder where the cursor was mapped on the plot,
# so I hovered over the x value I wanted for each edge if the "antineutron line"


# this one will not work because there is no "math operator for it"
# cut_mag = cut_W & (1.771 < p_p1.mag < 4.831) & (0.217 < p_pim.mag < 2.232) & (0.911 < p_e.mag < 3.363) & (1.771 < p_p2.mag < 4.831) 

plt.figure()
plt.hist2d(np.array(p_p1.mag[cut_mag]), np.array(MM_vec.M[cut_mag]), bins = 100, range = ((2.5, 7), (0, 2.5)), norm = 'log')
plt.xlabel('Momemtum magnitude of (detected) final state particles')
plt.ylabel('MM distro')
plt.title('MM distribution vs Momentum magnitude')
cuts_txt= (
        "Cuts: \n"
        r"1.771 < $|P_{p_1}|$ < 4.831GeV" "\n" 
        r"0.911 < $|P_{p_2}|$ < 2.730 GeV" "\n"
        r"0.217 < $|P_{\pi^-}|$ < 2.232 GeV" "\n"
        r"0.911 < $|P_{e'}|$  < 3.363 GeV" "\n"
        "3.1 < W < 4.09 GeV"

)
plt.text(
    5,1.5,
    cuts_txt    
)
plt.savefig('2d_histo_all_mom_mag.pdf')
plt.show()



#%%This is a histo to visualize combined magnitude cuts the signal we want
plt.figure()
plt.hist(MM_vec.M, bins = 20, range = (0.85, 1.15), histtype = 'step', color = 'black')
plt.hist(MM_vec.M[~cut_mag], bins = 20, range = (0.85, 1.15), color = 'green', alpha = 0.5)
plt.hist(MM_vec.M[cut_mag], bins = 20, range = (0.85, 1.15), color = 'blue', alpha = 0.5)
plt.xlabel('MM distribution (GeV)')
plt.ylabel('Counts')
plt.title('MM distribution as a result of momentum magnitude cuts')
plt.savefig('MM_all_mom_mag_.pdf')
plt.show()



# %%Beginning of Chi2pid cut
plt.figure()
plt.hist2d(np.array(p1_chi2pid), np.array(MM_vec.M), bins = 100, range = ((-5, 5), (0, 2.5)), norm = 'log')
plt.show()




# %%
cut_chi2pid = (np.abs(p1_chi2pid) <= 10) & cut_mag

plt.figure()
plt.hist(MM_vec.M, bins = 50, range = (0.85, 1.15), histtype = 'step', color = 'black')
plt.hist(MM_vec.M[~cut_chi2pid], bins = 50, range = (0.85, 1.15), color = 'green', alpha = 0.5)
plt.hist(MM_vec.M[cut_chi2pid], bins = 50, range = (0.85, 1.15), color = 'blue', alpha = 0.5)
plt.show()



#%% cut on lab frame angle distribution of missing mass
# use vec library to get theta

mass_cut = (MM_vec.M >= 0.85) & (MM_vec.M<=1.15)

p_nbar = vec.array({'px': MM_vec.px[mass_cut], "py": MM_vec.py[mass_cut], "pz": MM_vec.pz[mass_cut], "M": np.ones_like(MM_vec.px[mass_cut]) * mass_n})


# %%
plt.hist(np.rad2deg(p_nbar.theta), bins = 100)
# %%
