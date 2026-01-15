#Created on January 3 2026 01:14
#Author: Kevin Arias

# importing libraries
#%%
%matplotlib qt
import uproot
import numpy as np
import matplotlib.pyplot as plt
import vector as vec
import scipy as sp
import matplotlib as mpl
from upkit import Histo, Histo2D, RootAnalysis, Fit, tools

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


#%% missing mass distribution

# MM2 = E_mm**2 - (px_mm**2 + py_mm**2 + pz_mm**2)
MM_vec = p_beam + p_target - p_e - p_p1 - p_p2 - p_pim

plt.figure()
plt.hist(np.array(MM_vec.M), bins=200, range=(0.01, 2.5), histtype='step', color='black')
plt.axvline(mass_n, linestyle='--', label=f'Antineutron {mass_n}GeV' )
plt.legend()
plt.xlabel(r'$\bar{n}$ Missing Mass(GeV)')
plt.ylabel('Counts')
plt.title(r"Missing Mass Distribution for $ep \to e' p' p \pi^-$")
plt.legend()
plt.tight_layout()
plt.savefig('MM_no_cuts.pdf')
# plt.show()





#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
#%% This part is mainly to practice fitting and not necessary yet

# where did 0.02 come from?
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (60000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf))
bin_num = int((1.25-0.75)/(10e-3)) # 50 bins
bin_width = (1.25-0.75)/bin_num #  this is the bin width, 10e-3, because the resolution of the detector is 10-15 MeV

# this part fills histogram with MM values
# bin_content = counts in each bin (y positions)
# bin_edges = to help compute x positions, length = number of bins + 1
# _ = no idea why this is needed
def fit_dist(data, params, bounds, bin_num, fit_range=(0.75, 1.25)):
    bin_content, bin_edges, _ = plt.hist(data, bins = bin_num, range = fit_range)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    plt.close()
    plt.figure()



    # The FIT!!!
    # to get best fit parameter values
    # (function, x, y, parameters, bounds, uncertainty)
    fit_params, fit_cov = sp.optimize.curve_fit(
        gauss_poly4, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))

    # now to compute [[[signal yeild and it's uncertainty]]]!!!!!!!!!!

    # creating grid of x values for smooth fit curves
    x = np.linspace(fit_range[0], fit_range[1], 10000)

    # area of gaussian (height*width) divided by bin width to convert to number of events in histogram bins
    # used later
    fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/(bin_width)

    # annotating graph and then extracting
    plt.axvline(x=mass_n, color='red', linestyle='--', linewidth=2, label = "Neutron mass: 939.6 MeV")

    A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]

    # To show total detected events
    plt.axvline(x = 1, color = 'none', label = f"Total Events: {len(MM_vec.M):.3e}")

    what_sigma = 1
    # remember error propagation equation with partial derivatives
    yield_uncertainty = what_sigma*np.sqrt( 
        ((((np.sqrt(2*np.pi)*fit_params[2]) / bin_width) *A_uncertainty)**2) + 
        ((((np.sqrt(2*np.pi)*fit_params[0]) / bin_width) *sigma_uncertainty)**2) 
                                          )
    
    print(f'Relative Uncertainty in Yield = {yield_uncertainty/fit_yield}')

    plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')

    # statistical significance = a measure of significance of antineutron peak w.r.t. to background
    # It will be used to judge the usefullness of a cut. Each cut should increase the statistical significance
    # If a cut decreases siginificance, then that is a bad cut, if it stays the same then we keep it.

    # Error propagation for stat sig 
    db = np.sqrt(
        np.sqrt(np.diag(fit_cov))[3]**2 + 
        (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + 
        ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + 
        ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + 
        ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2
        ) 

    dsig = np.sqrt(
        (((1/np.sqrt(fit_params[0] + poly4(fit_params[1], *fit_params[3:])) - 
        (0.5*fit_params[0])/((fit_params[0] + poly4(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + 
        (((0.5*fit_params[0])/((fit_params[0] + poly4(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

    plt.axvline(
        x = 1, color = 'none', 
        label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.3g} $\pm$ {yield_uncertainty:.3g} (${what_sigma}\sigma$)")

    # S / sqrt( S + B )
    stat_sig = fit_params[0] / ( np.sqrt( fit_params[0] + poly4(fit_params[1], *fit_params[3:]) ) )
    print(f'Statistical Significance = {stat_sig:.4f}')
    plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.3f}")


    plt.hist(data, bins = bin_num, range=fit_range)
    plt.plot(x, gauss_poly4(x, *fit_params), color = 'red', label = "Background + Signal")
    plt.plot(x, poly4(x, *fit_params[3:]), linestyle = '--', label = "Background")
    plt.plot(x, gauss(x, *fit_params[:3]), color = 'cyan', label = "Signal")

    # might have to change this annotation later for each fitted MM plot
    plt.xlabel(r'$\bar{n}$ Missing Mass (GeV)')
    plt.ylabel('Counts/10 MeV')
    plt.legend()
    plt.tight_layout()
    # plt.show()

fit_dist(MM_vec.M, params, bounds, bin_num, fit_range=(0.75, 1.25))
plt.title('Gaussian + Background Fit to Antineutron MM peak')
plt.tight_layout()
plt.savefig('MM_no_cuts_fit.pdf')
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------


#%% invariant mass of final hadronic system (W)
p_W = p_beam + p_target - p_e
m_hadrons = 2*mass_p + mass_pim + mass_n
# another way could just be the sum of the final state hadrons (p' + p + pi + n)

plt.figure()
plt.hist(p_W.M, bins=100, range = (0, 5), histtype='step', color='k')
plt.axvline(m_hadrons, color = 'red', label=r'$2m_p + m_{\pi} + m_n$')
plt.legend()
plt.xlabel(r"$W_{(p' p \pi^- \bar{n})}$ (GeV)")
plt.ylabel('Counts')
plt.title('Hadronic Invariant Mass Spectrum')
plt.tight_layout()
plt.savefig('W_no_cuts.pdf')
# plt.show()



# %% This is a CUT!!! 
# Data is like a window and a cut is like a window cover

# cut_W = p_W.M > (2*mass_p + mass_pim + mass_n) = 2.9____
cut_W = (p_W.M > 3.3)  # you want a little more room

plt.figure()
plt.hist2d(np.array(p_W.M), np.array(MM_vec.M), bins = 100, range = ((0, 5), (0, 2.5)), norm = 'log')
plt.axvline(3.3, linestyle='--', color='red')
plt.xlabel(r"$W_{(p' p \pi^- \bar{n})}$ (GeV)")
plt.ylabel(r'$\bar{n}$ Missing Mass(GeV)')
plt.title(r'$\bar{n}_{MM}$ vs W Showing Electroproduction Band (no cuts)')
plt.text(
    3.2, 0.9,
    '                          ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)

plt.tight_layout()
plt.savefig('MM_vs_W.pdf')
# plt.show()

# Now with cut
plt.figure()
plt.hist2d(np.array(p_W.M[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 5), (0, 2.5)), norm = 'log')
plt.xlabel(r"$W_{(p' p \pi^- \bar{n})}$ (GeV)")
plt.ylabel(r'$\bar{n}$ Missing Mass (GeV)')
plt.title(r'$\bar{n}_{MM}$ vs W Showing Electroproduction Band (Threshold cut)')
plt.text(
    1, 1.5,
    r'Threshold cut: 3.3 GeV < W',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)
plt.text(
    3.2, 0.9,
    '                          ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)
plt.tight_layout()
plt.savefig('MM_vs_W_threshold_cut.pdf')
# plt.show()





# %% Missing Mass plot showing threshold cut

plt.figure()
# total (background + signal)
plt.hist(MM_vec.M, bins = 20, range = (0.85, 1.15), histtype = 'step', color = 'black', label='All events (No cuts)')

# background with cut
plt.hist(MM_vec.M[~cut_W], bins = 20, range = (0.85, 1.15), color = 'green', alpha = 0.5, label='Below threshold cut (background-like)')

# signal with cut --> [cut_W]
plt.hist(MM_vec.M[cut_W], bins = 20, range = (0.85, 1.15), color = 'blue', alpha = 0.5, label='Above threshold cut(signal-like))')

plt.legend()
plt.xlabel(r'$\bar{n}$ Missing Mass(GeV)')
plt.ylabel('Counts/10 MeV')
plt.title(r'Effect of $W$ Threshold Cut on Missing-Mass Spectrum')
plt.tight_layout()
plt.savefig('MM_threshold_cut.pdf')
# plt.show()

# these are now fitted with fit function (fit_distro) from before


# background + signal 
# plt.figure()
# fit_dist(MM_vec.M, params, bounds, bin_num, fit_range=(0.75, 1.25))
# plt.savefig('MM1.pdf')
# plt.show()

# signal
plt.figure()
fit_dist(MM_vec.M[cut_W], params, bounds, bin_num, fit_range=(0.75, 1.25))
plt.title('Fitted MM Spectrum After W Cut')
plt.tight_layout()
plt.savefig('MM2.pdf')
plt.show()

# just background
# plt.figure()
# fit_dist(MM_vec.M[~cut_W], params, bounds, bin_num, fit_range=(0.75, 1.25))
# plt.savefig('MM3.pdf')
# plt.show()



# %% These are 2d histograms of MM vs W with a momemtum magnitude cut and threshold cut with regards to different particles
plt.figure()
plt.hist2d(np.array(p_p1.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r'$|P_{p1}|$ (GeV)')
plt.ylabel('MM Distribution (GeV)')
plt.title(r'MM (with threshold cut) vs $|P_{p1}|$')
plt.text(
    1.612, 0.941,
    '                                                              ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)
plt.tight_layout()
plt.savefig('p1_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_p2.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r'$|P_{p2}|$ (GeV)')
plt.ylabel('MM Distribution (GeV)')
plt.title(r'MM (with threshold cut) vs $|P_{p2}|$')
plt.text(
    0.842, 0.915,
    '                                   ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)
plt.tight_layout()
plt.savefig('p2_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_e.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r"$|P_{e'}|$ (GeV)")
plt.ylabel('MM Distribution (GeV)')
plt.title(r"MM (with threshold cut) vs $|P_{e'}|$")
plt.text(
    0.753, 0.927,
    '                                            ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)

plt.tight_layout()
plt.savefig('e_mom_mag_threshold.pdf')
plt.show()

plt.figure()
plt.hist2d(np.array(p_pim.mag[cut_W]), np.array(MM_vec.M[cut_W]), bins = 100, range = ((0, 7), (0, 2.5)), norm = 'log')
plt.xlabel(r'$|P_{\pi^-}|$ (GeV)')
plt.ylabel('MM Distribution (GeV)')
plt.title(r'MM (with threshold cut) vs $|P_{\pi^-}|$')
plt.text(
    0.185, 0.927,
    '                                ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)
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
plt.hist(MM_vec.M, bins = 20, range = (0.85, 1.15), histtype = 'step', color = 'black', label='All events (No cuts)')
plt.hist(MM_vec.M[~cut_mag], bins = 20, range = (0.85, 1.15), color = 'green', alpha = 0.5, label='Below cuts (background-like)')
plt.hist(MM_vec.M[cut_mag], bins = 20, range = (0.85, 1.15), color = 'blue', alpha = 0.5, label='Above cuts cut(signal-like))')
plt.xlabel('MM distribution (GeV)')
plt.ylabel('Counts')
plt.title('MM Distribution as a Result of Momentum Magnitude Cuts')
plt.legend()
plt.tight_layout()
plt.savefig('MM_all_mom_mag_.pdf')
plt.show()

plt.figure()
fit_dist(MM_vec.M[cut_mag], params, bounds, bin_num, fit_range=(0.75, 1.25))
plt.title(r'Fitted MM Spectrum After Cuts $(W, |P|)$')
plt.tight_layout()
plt.savefig('all_mom_mag_fit.pdf')
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

plt.figure()
fit_dist(MM_vec.M[cut_chi2pid], params, bounds, bin_num, fit_range=(0.75, 1.25))
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

# %% this is the start of a delta time cut

# plt.figure()
# plt.hist2d(np.array(p_p1.mag), np.array(dt_p1), bins=(np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range=((0,6),(-5,5)), norm = mpl.colors.LogNorm())
# plt.xlabel(r"$|P_{p1}|$(GeV)")
# plt.ylabel(r"$\Delta t_{p1}$ (ns)")
# plt.title(r'$\Delta t_{p1}$ vs $|P_{p1}|$')
# plt.savefig('dt_p1.pdf')
# plt.show()


# using for loop instead
particles = ['p1', 'p2', 'pim', 'e']
momenta  = [p_p1.mag,  p_p2.mag,  p_pim.mag,  p_e.mag]
dts      = [dt_p1,     dt_p2,     dt_pim,     dt_e]

for name, p_mag, dt in zip(particles, momenta, dts):
    plt.figure()
    plt.hist2d(np.array(p_mag), np.array(dt),
               bins=(np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), 
               range=((0, 6), (-5, 5)), norm=mpl.colors.LogNorm())
    plt.xlabel(rf'$|P_{{{name}}}|$ (GeV)')
    plt.ylabel(rf'$\Delta t_{{{name}}}$ (ns)')
    plt.title(rf'$\Delta t$ vs $|P_{{{name}}}|$')
    # plt.colorbar(label='Counts')
    plt.tight_layout()
    plt.savefig(f'dt_{name}_vs_p_{name}.pdf')
    plt.show()

dt_windows = {
    'p1':  (-0.5, 0.75),
    'p2':  (-0.9, 1.0),
    'pim': (-0.5, 0.75),
    'e':   (-0.5, 0.35),
}

cuts_dt = {}

for name, dt in zip(particles, dts):
    lo, hi = dt_windows[name]
    cuts_dt[name] = (dt > lo) & (dt < hi)


# combinning in one cut 
cut_dt_all = cuts_dt['p1'] & cuts_dt['p2'] & cuts_dt['pim'] & cuts_dt['pim'] & cuts_dt['e']
# combinning cuts
cut_all = cut_chi2pid & cut_dt_all


#%%
plt.figure()
plt.hist(MM_vec.M, bins = 30, range = (0.85, 1.15), histtype = 'step', color = 'black')
plt.hist(MM_vec.M[~cut_all], bins = 30, range = (0.85, 1.15), color = 'green', alpha = 0.5)
plt.hist(MM_vec.M[cut_all], bins = 30, range = (0.85, 1.15), color = 'blue', alpha = 0.5)
plt.xlabel('Missing Mass (GeV)')
plt.ylabel('Counts')
plt.title(r'MM with $\Delta$t cuts')
plt.tight_layout()
# plt.savefig('.pdf')
plt.show()


#%%
plt.figure()
fit_dist(MM_vec.M[cut_all], params, bounds, bin_num, fit_range=(0.75, 1.25))
plt.show()


#%%
# 1) Choose momentum bins.
# 2) For each bin, select events in that momentum range.
# 3) Plot (and optionally fit) the 1D MM distribution for that slice.

p_slice = p_p1.mag
mm = MM_vec.M

# define momemtum bins
p_bins = np.arange(0, 6, 0.5)

# looping over momemtum slices and making 1d MM histos
for i in range(len(p_bins) - 1):
    p_min = p_bins[i]
    p_max = p_bins[i+1]

    slice_mask = (p_slice >= p_min) & (p_slice < p_max) & cut_all
    mm_slice = mm[slice_mask]

    # skip very low-stat slices
    if len(mm_slice) < 30:   # choose a threshold (20–50 is typical)
        continue

    
    plt.figure()
    fit_dist(mm_slice, params, bounds, bin_num, fit_range=(0.75, 1.25))
    # plt.savefig(f'MM_p1_slice_{p_min:.2f}_{p_max:.2f}.pdf')
    plt.show()



# %%
# Available keys: 'deltaTime_pim', 
# 'deltaTime_p1', 'deltaTime_p2', 'deltaTime_e', 
# 'beta_pim', 'beta_p1', 'beta_p2', 'theta_pim', 
# 'beta_e', 'theta_p1', 'theta_p2', 'status_pim', 
# 'theta_e', 'pid_pim', 'phi_pim', 'betafromP_pim'...


###Defining momentum mag and delta time for pip
# wmask = (W > (W_thry + 0.39)) & (data.MM > 0) & (data.MM < 3)
# p3_pip, dt_pip = np.array(data['P_mag_pip'])[wmask], np.array(data['deltaTime_pip'])[wmask]

# %%
