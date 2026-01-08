#Created on January 3 2026 01:14
#Author: Kevin Arias

# importing libraries
#%%
# %matplotlib qt.  #This make plots in a new window
import uproot
import numpy as np
import matplotlib.pyplot as plt
import vector as vec
import scipy as sp
import matplotlib as mpl

# constants
mass_p = 0.938272088  # GeV/c^2 for proton
mass_pim = 0.13957     # GeV/c^2 for pion
mass_pip = 0.13957     # GeV/c^2 for pion
mass_k = 0.49367     # GeV/c^2 for kaon
mass_e = 0.000511     # GeV/c^2 for electron
E_beam = 10.2         # GeV (assuming beam energy of 10.6 GeV)
mass_n = .939565      # GeV/c^2 for neutron

# defining fitting functions to be used throughout this script
def gauss(x, A, mu, sigma):
    gauss = A*np.exp( -(x - mu)**2 / (2*sigma**2) )
    return gauss

def poly4(x, a, b, c, d, e):
    poly4 = a + b*x + c*x**2 + d*x**3 + e*x**4
    return poly4

def expo(x, A, B):
    expo = A*np.exp(B*x)
    return expo

def gauss_poly4(x, A, mu, sigma, a, b, c, d, e):
    gauss_poly4 = gauss(x, A, mu, sigma) + poly4(x, a, b, c, d, e)
    return gauss_poly4

def expo_poly4(x, A ,B, a, b, c, d, e):
    expo_poly4 = expo(x, A, B) + poly4(x, a, b, c, d, e)
    return expo_poly4


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
#%% Definning 4 momenta
p_beam = vec.obj(px = 0, py = 0, pz = 10.2, E = 10.2)
p_target = vec.obj(px = 0, py = 0, pz = 0, M = 0.938272088)

p_e = vec.array({'px': px_e, "py": py_e, "pz": pz_e, "M": np.ones_like(px_e) * mass_e})
p_p1 = vec.array({'px': px_p1, "py": py_p1, "pz": pz_p1, "M": np.ones_like(px_p1) * mass_p})
p_p2 = vec.array({'px': px_p2, "py": py_p2, "pz": pz_p2, "M": np.ones_like(px_p2) * mass_p})
p_pim = vec.array({'px': px_pim, "py": py_pim, "pz": pz_pim, "M": np.ones_like(px_pim) * mass_pim})


#%% missing mass method

# MM2 = E_mm**2 - (px_mm**2 + py_mm**2 + pz_mm**2)
MM_vec = p_beam + p_target - p_e - p_p1 - p_p2 - p_pim
plt.figure()
plt.hist(np.array(MM_vec.M), bins=200, range=(0.01, 2.5), histtype='step', color='black')
plt.axvline(mass_n, linestyle='--', label=f'Antineutron {mass_n}GeV' )
plt.legend()
plt.xlabel('Missing Mass(GeV)')
plt.ylabel('Counts')
plt.title('Missing Mass Distribution (no cuts)')


plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(MM_vec.M, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

# fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly4, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))
sigma = np.sqrt(bin_content)
sigma[sigma == 0] = 1.0  # or mask these out

mask = bin_content > 0
fit_params, fit_cov = sp.optimize.curve_fit(
    gauss_poly4,
    bin_centers[mask],
    bin_content[mask],
    p0=params,
    bounds=bounds,
    sigma=sigma[mask],
)


x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
#plt.close()
#plt.figure()
plt.axvline(x=mass_n, color='red', linestyle='--', linewidth=2, label = "Neutron mass: 939.6 MeV")
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(MM_vec.M)}")
plt.tight_layout()
plt.savefig('MM_no_cuts.pdf')
plt.show()


#%% invariant mass of final hadronic system (W)

p_W = p_beam + p_target - p_e

plt.figure()
plt.hist(p_W.M, bins=100, range = (0, 5), histtype='step', color='k')
plt.axvline(2*mass_p + mass_pim + mass_n, color = 'red', label=r'$2m_p + m_{\pi} + m_n$')
plt.legend()
plt.xlabel('W (no cuts)')
plt.ylabel('Counts')
plt.title('Hadronic Invariant Mass Distribution(no cuts)')
plt.tight_layout()
plt.savefig('W_no_cuts_ZOOM_OUT.pdf')
plt.show()



# %% This is a CUT!!! 
# Data is like a window and a cut is like a window cover

# cut_W = p_W.M > (2*mass_p + mass_pim + mass_n) = 2.9____
cut_W = (p_W.M > 3.3)  # you want a little more room

plt.figure()
plt.hist2d(np.array(p_W.M), np.array(MM_vec.M), bins = 100, range = ((0, 5), (0, 2.5)), norm = 'log')
plt.axvline(3.3, linestyle='--', color='red')
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
    1, 1.5,
    'Threshold cut: 3.3 < W',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)
plt.text(
    3.2, 0.9,
    '                          ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)
plt.tight_layout()
plt.savefig('MM_vs_W_threshold_cut.pdf')
plt.show()




# %% Missing Mass plot showing threshold cut
plt.figure()
plt.hist(MM_vec.M, bins = 20, range = (0.85, 1.15), histtype = 'step', color = 'black', label='Total MM')
plt.hist(MM_vec.M[~cut_W], bins = 20, range = (0.85, 1.15), color = 'green', alpha = 0.5, label='Background')
plt.hist(MM_vec.M[cut_W], bins = 20, range = (0.85, 1.15), color = 'blue', alpha = 0.5, label='Signal')
plt.legend()
plt.xlabel('Missing Mass(GeV)')
plt.ylabel('Counts')
plt.title('Missing Mass distribution showing threshold cut')
plt.tight_layout()
plt.savefig('MM_threshold_cut.pdf')
plt.show()





# %% These are 2d histograms of MM vs W with a momemtum magnitude cut and threshold cut with reguards to different particles
plt.figure()
plt.hist2d(np.array(p_p1.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel('Proton 1 magnitude with threshold cut(GeV)')
plt.ylabel('MM with threshold cut')
plt.title('proton 1 momemtum magnitude vs W (with threshold cut)')
plt.tight_layout()
plt.savefig('p1_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_p2.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel('Proton 2 magnitude with threshold cut(GeV)')
plt.ylabel('MM with threshold cut')
plt.title('proton 2 momemtum magnitude vs W (with threshold cut)')
plt.tight_layout()
plt.savefig('p2_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_e.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r"$e\text{'}$ momentum magnitude with threshold cut(GeV)")
plt.ylabel('MM with threshold cut')
plt.title(r"$e\text{'}$ momemtum magnitude vs W (with threshold cut)")
plt.tight_layout()
plt.savefig('e_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_pim.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r"$\pi^-$ momentum magnitude with threshold cut(GeV)")
plt.ylabel('MM with threshold cut')
plt.title(r"$\pi^-$ momemtum magnitude vs W (with threshold cut)")
plt.tight_layout()
plt.savefig('pi_mom_mag_threshold.pdf')
plt.show()



# %% This is a momentum magnitude cut using all particles

cut_mag = cut_W & (p_p1.mag > 1.771) & (p_pim.mag < 2.232) & (p_e.mag > 0.911) & (p_p2.mag < 2.730)
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
        r"1.771 GeV < $|P_{p_1}|$" "\n" 
        r"0.911 GeV < $|P_{p_2}|$" "\n"
        r"0.217 GeV < $|P_{\pi^-}|$" "\n"
        r"0.911 GeV < $|P_{e'}|$" "\n"
        "3.3 GeV< W"

)
plt.text(
    5,1.5,
    cuts_txt    
)
plt.tight_layout()
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
plt.tight_layout()
plt.savefig('MM_all_mom_mag_.pdf')
plt.show()



# %%Beginning of Chi2pid cut
# Chi2PID = not like chi squared, 

plt.hist2d(np.array(p1_chi2pid), np.array(MM_vec.M), bins = 100, range = ((-5, 5), (0, 2.5)), norm = 'log')
plt.xlabel(r"$\chi^2_{PID}$")
plt.ylabel('Counts')
plt.title(r"$\chi^2_{PID}$")
plt.tight_layout()
plt.savefig('Chi2PID.pdf')
plt.show()




# %% MM plot showing Chi2PID cut
cut_chi2pid = (np.abs(p1_chi2pid) <= 10) & cut_mag

plt.figure()
plt.hist(MM_vec.M, bins = 30, range = (0.85, 1.15), histtype = 'step', color = 'black')
plt.hist(MM_vec.M[~cut_chi2pid], bins = 30, range = (0.85, 1.15), color = 'green', alpha = 0.5)
plt.hist(MM_vec.M[cut_chi2pid], bins = 30, range = (0.85, 1.15), color = 'blue', alpha = 0.5)
plt.xlabel('Missing Mass (GeV)')
plt.ylabel('Counts')
plt.title(r'MM with $\chi^2_{PID}$ cut')
plt.tight_layout()
plt.savefig('MM_Chi2PID.pdf')
plt.show()


#%% cut on lab frame angle distribution of missing mass
# use vec library to get theta
mass_cut = (MM_vec.M >= 0.85) & (MM_vec.M<=1.15)
p_nbar = vec.array({'px': MM_vec.px[mass_cut], "py": MM_vec.py[mass_cut], "pz": MM_vec.pz[mass_cut], "M": np.ones_like(MM_vec.px[mass_cut]) * mass_n})


# %% This is a histogram of the antineutron theta angle that is calculated from four vector calculation
# this calculation is done automatically thanks to the vector library
plt.hist(np.rad2deg(p_nbar.theta), bins = 100)
plt.xlabel(r'$\theta_{\bar{n}}$')
plt.ylabel('Counts')
plt.title('Lab Frame Angle Distribution of MM')
plt.tight_layout()
plt.savefig('theta_anti_N.pdf')
# %%
