# #Created on January 3 2026 01:14
#Author: Kevin Arias

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
file = uproot.open('Pppim_eFT_all.root')
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
plt.savefig('FT_MM_no_cuts.pdf')
# plt.show()
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
    plt.figure(figsize=(9,7))

     # mask out empty bins so sigma is finite and nonzero
    mask = bin_content > 0
    bin_content_fit = bin_content[mask]
    bin_centers_fit = bin_centers[mask]
    sigma_fit = np.sqrt(bin_content_fit)

    # The FIT!!!
    # to get best fit parameter values
    # (function, x, y, parameters, bounds, uncertainty)
    # fit_params, fit_cov = sp.optimize.curve_fit(
    #     gauss_poly4, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))

    fit_params, fit_cov = sp.optimize.curve_fit(
    gauss_poly4,
    bin_centers_fit,
    bin_content_fit,
    p0=params,
    bounds=bounds,
    sigma=sigma_fit,
    )

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
        label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {int(fit_yield)} $\pm$ {yield_uncertainty:.3g} (${what_sigma}\sigma$)")

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
plt.savefig('FT_MM_no_cuts_fit.pdf')

# %%
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
plt.savefig('FT_W_no_cuts.pdf')
# plt.show()

#%% 2D Histo of W and MM to see antineutron cut
Histo2D(p_W.M, MM_vec.M, range = ((3, 4.5), (.5, 1.5)), bins = [1000, 1000], norm = 'log')
plt.axvline(3.6, linestyle='--', color='red')
plt.xlabel(r"$W_{(p' p \pi^- \bar{n})}$ (GeV)")
plt.ylabel(r'$\bar{n}$ Missing Mass(GeV)')
plt.title(r'$\bar{n}_{MM}$ vs W Showing Electroproduction Band (no cuts)')
plt.text(
    3.6, 0.910,
    '                                                    \n                           \n                           ',
    bbox=dict(boxstyle='round', facecolor='none', alpha=1)        
)

plt.tight_layout()
plt.savefig('FT_MM_vs_W.pdf')

# %%
cut = (p_W.M > 3.6)

# try this too (p_W.M < 4.4) & 
h2_W_MM = Histo2D(p_W.M[cut], MM_vec.M[cut], range = ((3.2, 4.5), (.5, 1.5)), bins = [100, 100], norm = 'log')
h2_W_MM.show_hists(xlabel ='W', ylabel='MM')
plt.tight_layout()
plt.savefig('FT_MM_vs_W_threshold_cut.pdf')
# plt.show()

# %%

fig, ax = plt.subplots()
Histo(MM_vec.M, bins = 100, range = (0, 2.5), ax = ax)
Histo(MM_vec.M[cut], bins = 100, range =(0, 2.5), ax = ax)
Histo(MM_vec.M[~cut], bins = 100, range =(0, 2.5), ax = ax)
# %%
