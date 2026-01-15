#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:17:12 2025

@author: ejmorell
"""

### This purpose of this script is to load raw experimental data, perform 
###various cuts to reduce background contributions, and save the result as 
### a ROOT file.

### It is quite messy as there was a lot of trial and error. If you are unsure where 
### something comes from or what it means, my honors thesis is a good resource to site.
### Otherwise feel free to reach out and I will do my best to help.

#Necessary libraries
import uproot
import vector as vec
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
import awkward as ak

#Some constants
mass_p = 0.938272088
mass_e = 0.000511
mass_pip = 0.1396
mass_n = 0.93956


### Fitting functions to be used later
def gauss_fit(x, amp, mean, sigma):
    gauss = amp*np.exp(-(x-mean)**2/(2*sigma**2))
    
    return gauss

def poly_fit(x, a, b, c, d, e):
    poly = a+b*x**1+c*x**2+d*x**3+e*x**4
    
    return poly

def gauss_poly_fit(x, amp, mean, sigma, a, b, c, d, e):
    gauss_poly = gauss_fit(x, amp, mean, sigma) + poly_fit(x, a, b, c, d, e)
    
    return gauss_poly

def exp_fit(x, amp, A):
    exp = amp*np.exp(A*(x))
    
    return exp

def exp_poly_fit(x, amp, A, a, b, c, d, e):
    exp_poly = exp_fit(x, amp, A) + poly_fit(x, a, b, c, d, e)
    
    return exp_poly


#%%
### Loading ROOT file and tree branches

file = uproot.open('/Users/ejmorell/Work/Nuclear Internship/Neutron Analysis/Data Files/Ppbar_FD.root')

tree = file['Individual']

data = tree.arrays()


#%%
### Defining variables from branches

px_p, py_p, pz_p = data['px_p'], data['py_p'], data['pz_p']
px_pb, py_pb, pz_pb = data['px_pb'], data['py_pb'], data['pz_pb']
px_pip, py_pip, pz_pip = data['px_pip'], data['py_pip'], data['pz_pip']
px_e, py_e, pz_e = data['px_e'], data['py_e'], data['pz_e'], 

### Defining 4-momenta

p_target = vec.obj(px = 0, py = 0, pz = 0, E = mass_p)
p_beam = vec.obj(px = 0, py = 0, pz = 10.2, E = 10.2)
p_e = vec.array({'px': px_e, 'py': py_e, 'pz': pz_e, 'M': np.ones_like(pz_e)*mass_e})
p_p = vec.array({'px': px_p, 'py': py_p, 'pz': pz_p, 'M': np.ones_like(pz_e)*mass_p})
p_pb = vec.array({'px': px_pb, 'py': py_pb, 'pz': pz_pb, 'M': np.ones_like(pz_e)*mass_p})
p_pip = vec.array({'px': px_pip, 'py': py_pip, 'pz': pz_pip, 'M': np.ones_like(pz_e)*mass_pip})

#%%

### Defining MM(p pb pip e)
p_miss = p_beam + p_target - p_p - p_pb - p_pip - p_e
mass_miss = p_miss.M

### Histogram of uncut missing mass

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mass_miss, bins = bins, range = (0,3))
plt.axvline(x=mass_n, color='red', linestyle='--', linewidth=2, label = "Neutron mass: 939.6 MeV")
plt.legend()
plt.xlabel(r'$m_{miss}$ $[GeV/c^2]$', fontsize = 12.5)
plt.ylabel("Counts / 15 MeV", fontsize = 12.5)
plt.title('Electron in Forward Detector', fontsize = 17.5)

#%%

### This section zooms in on the area of interest. The commented lines contain
### commands to fit the data, though it is not really necessary at this point.

plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mass_miss, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
#plt.close()
#plt.figure()
plt.axvline(x=mass_n, color='red', linestyle='--', linewidth=2, label = "Neutron mass: 939.6 MeV")
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(data.MM)}")

### You will see this statistical significance parameter a lot. It is basically
### just a measure of the significance of our neutron peak with respect to the
### the background under the peak. I use it to determine whether a cut is useful
### or not. Ideally statistical significance should increase with each cut. If
### it decreases it is a bad cut and if it stays the same it is a useless cut,
### so no need to throw out that data.
stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

#plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
#plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
#plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
#plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
#plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'$m_{miss}$ $[GeV/c^2]$', fontsize = 12.5)
plt.ylabel("Counts / 15 MeV", fontsize = 12.5)
plt.title('Electron in Forward Detector', fontsize = 17.5)

#%%
### The first cut is on the mass of the final hadronic system. Energy-momentum
### conservation tells us the 4-momentum of the final hadronic system is the 
### initial state minus the leptons in final state (in this case electron)

### Mass of hadronic system, W
W = (p_beam + p_target - p_e).M

### Histogram showing W and threshold
plt.figure()
bins = np.arange(0,5 + 0.015, 0.015)
plt.hist(W, bins = bins, range = (0,5))
plt.axvline(x=(2*mass_p + mass_pip + mass_n), color='red', linestyle='--', linewidth=2, label = "W Mass Threshold = 2.956 GeV")
plt.xlabel("Mass (GeV)")
plt.title(r"W Mass")
plt.ylabel("Counts / 15MeV")
plt.legend()

#%%
### Another way to define the mass of hadronic system is by of course directly 
### adding the masses of the final state hadrons. In the case of final state 
### hadrons that are stationary we get:
W_thry = 2*mass_p + mass_pip + mass_n

### Showing original histogram, cut histogram, and anti-cut (the data that we removed)
### overlayed for vizualization.
plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(data.MM, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(data.MM[W < (W_thry + 0.39)], bins = bins, range = (0, 3), alpha = 0.3, label = "W Mass Anti-Cut")
plt.hist(data.MM[W > (W_thry + 0.39)], bins = bins, range = (0, 3), alpha = 0.3, label = "W Mass Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title("Results of W-Mass cut (eFD)")

### Since we know that the final state particles need not be stationary, we conclude
### that the mass of hadronic system of our final state hadrons must be greater than
### or equal to the minumum hadronic mass given by stationary final state hadrons. 
wmask = (W > (W_thry + 0.39)) & (data.MM > 0) & (data.MM < 3)

wcut_mm = data.MM[wmask]

#%%

###Fitting W-cut MM and graphing

plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(wcut_mm, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(data.MM[wmask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.title('Electron in FD')
plt.ylabel("Counts / 15 MeV")


#%%
### The next cut is on delta time. We plot it as a 2D histogram with momentum 
### because there is a dependence.

###Defining momentum mag and delta time for pip
p3_pip, dt_pip = np.array(data['P_mag_pip'])[wmask], np.array(data['deltaTime_pip'])[wmask]

### 2d hist showing momentum dependence of delta time
plt.figure()
dtime = plt.hist2d(np.array(p3_pip), np.array(dt_pip), bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')

H, xedges, yedges, img = dtime

#%%

### The next section is a bit convoluted. We slice our 2d histgram into individual
### 1D histgrams for a given momentum. then fit each woth a gaussian. Then,
### we use this gaussian to determine some cutoff point, in this case 5 sigma.
### aThese +/- 5sigma points are then plotted on the original 2D histgram and fit
### with exponentials. These exponentials determione where we are cutting our data.

### Defining arrays for fitting parameters from 1D histograms
A_fit = np.empty(len(H[:,:]))
mu_fit = np.empty_like(A_fit)
sigma_fit = np.empty_like(A_fit)


bin_centers = ((yedges[:-1] + yedges[1:])/2)
#bin_centers = pre_bin_centers[(pre_bin_centers >= -1) & (pre_bin_centers <= 1)]

###Slicing 2D histogram into 1D dt slices for a given momentum, then fitting with gaussian
for i in range(len(H[:,:])):
    sliced_hist = H[i, :]
    
    if np.sum(sliced_hist) == 0:
        continue
    
    params = [50000, 0, 0.986]
    bounds = ((0, -1, 0), (100000, 1, 5))
    
    try:
        fit_params, fit_cov = sp.optimize.curve_fit(gauss_fit, bin_centers, sliced_hist, p0 = params, bounds = bounds)
        A_fit[i], mu_fit[i], sigma_fit[i] = fit_params[:3]
    except RuntimeError:
        print(f"Gaussian fit failed for column {i}")
        continue  # Skip to next column if fit fails
    
    # plt.figure()
    # plt.bar(bin_centers, sliced_hist, width = yedges[1] - yedges[0])
    # x = np.linspace(-1.5, 1.5, 1000)
    # fit = gauss_fit(x, *fit_params)
    # plt.plot(x, fit, color = 'red')
        
### Plotting a number of sigma values from our fits on 2D hist

s = 5 ### # of sigma i am cutting at
xbin_centers = ((xedges[:-1] + xedges[1:])/2)
cut_sigma_fit = sigma_fit[xbin_centers > 0.2]
sigma_vals = np.append(s*cut_sigma_fit, -s*cut_sigma_fit)

plt.figure()
dtime = plt.hist2d(np.array(p3_pip), np.array(dt_pip), bins = (np.arange(0, 6, 0.05), np.arange(-1.5, 1.5, 0.05)), range = ((0, 6), (-1.5, 1.5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.scatter(np.append(xbin_centers[xbin_centers > 0.2], xbin_centers[xbin_centers > 0.2]), sigma_vals, color = 'red', s = 10)

        
        #%%

plt.figure()
dtime = plt.hist2d(np.array(p3_pip), np.array(dt_pip), bins = (np.arange(0, 6, 0.05), np.arange(-1.5, 1.5, 0.05)), range = ((0, 6), (-1.5, 1.5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.scatter(np.append(xbin_centers[xbin_centers > 0.2], xbin_centers[xbin_centers > 0.2]), sigma_vals, color = 'red', s = 10)

###Fitting the lower end momenta
# params = [1, 1, 1, 1, 1]
# bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf))

# x1 = np.linspace(0.2, 0.6, 10000)

# fit_params11, fit_cov = sp.optimize.curve_fit(poly_fit, xbin_centers[(xbin_centers > 0.2) & (xbin_centers < 0.6)], s*sigma_fit[(xbin_centers > 0.2) & (xbin_centers < 0.6)], p0 = params, bounds = bounds)
# fit11 = poly_fit(x1, *fit_params11)
# plt.plot(x1, fit11, color = 'black')

# params = [1, 1, 1, 1, 1]
# bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf))

# fit_params12, fit_cov = sp.optimize.curve_fit(poly_fit, xbin_centers[(xbin_centers > 0.2) & (xbin_centers < 0.6)], -s*sigma_fit[(xbin_centers > 0.2) & (xbin_centers < 0.6)], p0 = params, bounds = bounds)
# fit12 = poly_fit(x1, *fit_params12)
# plt.plot(x1, fit12, color = 'black')

###Fitting data points with exponentials.

params = [1, -1, 1, 1, 1, 1, 1]
bounds = ((0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf, np.inf, np.inf, np.inf, np.inf))

x2 = np.linspace(0, 6, 10000)

fit_params21, fit_cov = sp.optimize.curve_fit(exp_poly_fit, xbin_centers[(xbin_centers > 0.6)], s*sigma_fit[(xbin_centers > 0.6)], p0 = params, bounds = bounds)
fit21 = exp_poly_fit(x2, *fit_params21)
plt.plot(x2, fit21, color = 'black')

params = [-1, -1, 1, 1, 1, 1, 1]
bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (0, 0, np.inf, np.inf, np.inf, np.inf, np.inf))

fit_params22, fit_cov = sp.optimize.curve_fit(exp_poly_fit, xbin_centers[xbin_centers > 0.6], -s*sigma_fit[(xbin_centers > 0.6)], p0 = params, bounds = bounds)
fit22 = exp_poly_fit(x2, *fit_params22)
plt.plot(x2, fit22, color = 'black')

#%%

###Plotting hist with expoential, but without points
plt.figure()
dtime = plt.hist2d(np.array(p3_pip), np.array(dt_pip), bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
# plt.plot(x1, fit11, color = 'red')
# plt.plot(x1, fit12, color = 'red')
plt.plot(x2, fit21, color = 'red')
plt.plot(x2, fit22, color = 'red')

#%%

### Creating mask array to remove values from cut

# mask11 = dt_pip[p3_pip < 0.6] < poly_fit(p3_pip[p3_pip < 0.6], *fit_params11)
# mask12 = dt_pip[p3_pip < 0.6] > poly_fit(p3_pip[p3_pip < 0.6], *fit_params12)
mask21 = dt_pip < exp_poly_fit(p3_pip, *fit_params21)
mask22 = dt_pip > exp_poly_fit(p3_pip, *fit_params22)

mask = np.zeros_like(dt_pip, dtype=bool)

# mask[p3_pip < 0.6] = mask11 & mask12
mask = mask21 & mask22
x2 = np.linspace(0.6, 6, 10000)

### Plotting cut 2d hist
plt.figure()
plt.scatter(np.append(xbin_centers[xbin_centers > 0.2], xbin_centers[xbin_centers > 0.2]), sigma_vals, color = 'red', s = 10)
dtime = plt.hist2d(np.array(p3_pip)[mask], np.array(dt_pip)[mask], bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
# plt.plot(x1, fit11, color = 'red')
# plt.plot(x1, fit12, color = 'red')
plt.plot(x2, fit21, color = 'red')
plt.plot(x2, fit22, color = 'red')



#%% Plotting cut results
plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(wcut_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(wcut_mm[~mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(wcut_mm[mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

#%%
###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(wcut_mm[mask], bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

dtpip_mm = wcut_mm[mask]

#%%
### Same thing this time for antiproton. It is commented because it was found
### to be ineffective

p3_pb, dt_pb = np.array(data['P_mag_pb'])[wmask][mask], np.array(data['deltaTime_pb'])[wmask][mask]

 ### 2d hist showing momentum dependence of delta time
plt.figure()
dtime = plt.hist2d(np.array(p3_pb), np.array(dt_pb), bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\bar{p}$')

# H, xedges, yedges, img = dtime

# plt.figure()
# plt.hist(np.array(dt_pb), bins =  np.arange(-2, 2, 0.05), range = ((-2, 2)))
# plt.xlabel('Momentum Magnitude (GeV)')
# plt.ylabel('Delta Time (ns)')
# plt.title(r'$\Delta t$ vs Momentum for $\bar{p}$')

# params = [4000, 0, 0.02, 1, 1, 1, 1, 1]
# bounds = ((0, -10, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 10, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

# bin_content, bin_edges, _ = plt.hist(np.array(dt_pb), bins = np.arange(-2, 2, 0.05), range = (-2, 2))

# bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

# fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma  = np.sqrt(bin_content))

# x = np.linspace(-2, 2, 10000)
# fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015

# plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
# plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
# plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
# plt.axvline(x = 1, color = 'none', label = f"Mean = {fit_params[1]:.3f}\nSigma = {fit_params[2]:.3f}\nYield = {fit_yield:.0f}")
# plt.legend()

# tpb_mask = (dt_pb < 3*fit_params[2]) & (dt_pb > -3*fit_params[2])

# #%%
# plt.figure()
# bins = np.arange(0, 3 + 0.015, 0.015)
# plt.hist(wcut_mm[mask], bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
# plt.hist(wcut_mm[mask][~tpb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
# plt.hist(wcut_mm[mask][tpb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
# plt.legend()
# plt.xlabel("Mass (GeV)")
# plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

###I keep this variable for the sake of efficiency because it is used in later analysis 
###but i remved the cut. You can ignore it.
tpb_mm = wcut_mm[mask]#[tpb_mask]

# ###Fitting and plotting results
# plt.figure()
# params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
# bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

# bins = np.arange(0.6, 1.5 + 0.015, 0.015)
# bin_content, bin_edges, _ = plt.hist((tpb_mm), bins = bins, range = (0.6, 1.5))

# bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

# fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma =  np.sqrt(bin_content))

# x = np.linspace(0.6, 1.5, 10000)
# fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015

# plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
# plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
# plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
# plt.axvline(x = 1, color = 'none', label = f"Mean = {fit_params[1]:.3f}\nSigma = {fit_params[2]:.3f}\nYield = {fit_yield:.0f}")
# plt.legend()
#%%
###Delte time cut for proton.
p3_p, dt_p = np.array(data['P_mag_p'])[wmask][mask], np.array(data['deltaTime_p'])[wmask][mask]

### 2d hist showing momentum dependence of delta time
plt.figure()
dtime = plt.hist2d(np.array(p3_p), np.array(dt_p), bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $p$')

H, xedges, yedges, img = dtime

#%%

### Defining arrays for fitting parameters from 1D histograms
A_fit = np.empty(len(H[:,:]))
mu_fit = np.empty_like(A_fit)
sigma_fit = np.empty_like(A_fit)


bin_centers = ((yedges[:-1] + yedges[1:])/2)
#bin_centers = pre_bin_centers[(pre_bin_centers >= -1) & (pre_bin_centers <= 1)]

###Slicing 2D histogram into 1D dt slices for a given momentum, then fitting with gaussian
for i in range(len(H[:,:])):
    sliced_hist = H[i, :]
    
    if np.sum(sliced_hist) == 0:
        continue
    
    params = [50000, 0, 0.986]
    bounds = ((0, -1, 0), (100000, 1, 5))
    
    try:
        fit_params, fit_cov = sp.optimize.curve_fit(gauss_fit, bin_centers, sliced_hist, p0 = params, bounds = bounds)
        A_fit[i], mu_fit[i], sigma_fit[i] = fit_params[:3]
    except RuntimeError:
        print(f"Gaussian fit failed for column {i}")
        continue  # Skip to next column if fit fails
    
    # plt.figure()
    # plt.bar(bin_centers, sliced_hist, width = yedges[1] - yedges[0])
    # x = np.linspace(-1.5, 1.5, 1000)
    # fit = gauss_fit(x, *fit_params)
    # plt.plot(x, fit, color = 'red')
    
    ### Plotting a number of sigma values from our fits on 2D hist

s = 5 ### # of sigma i am cutting at
xbin_centers = ((xedges[:-1] + xedges[1:])/2)
cut_sigma_fit = sigma_fit[xbin_centers > 0.7]
sigma_vals = np.append(s*cut_sigma_fit, -s*cut_sigma_fit)

plt.figure()
dtime = plt.hist2d(np.array(p3_p), np.array(dt_p), bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.scatter(np.append(xbin_centers[xbin_centers > 0.7], xbin_centers[xbin_centers > 0.7]), sigma_vals, color = 'red', s = 10)

#%%

plt.figure()
dtime = plt.hist2d(np.array(p3_p), np.array(dt_p), bins = (np.arange(0, 6, 0.05), np.arange(-3, 3, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.scatter(np.append(xbin_centers[xbin_centers > 0.7], xbin_centers[xbin_centers > 0.7]), sigma_vals, color = 'red', s = 10)

params = [1, -1, 1, 1, 1, 1, 1]
bounds = ((0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf, np.inf, np.inf, np.inf, np.inf))

x2 = np.linspace(0.2, 6, 10000)

fit_params21, fit_cov = sp.optimize.curve_fit(exp_poly_fit, xbin_centers[(xbin_centers > 0.7)], s*sigma_fit[(xbin_centers > 0.7)], p0 = params, bounds = bounds)
fit21 = exp_poly_fit(x2, *fit_params21)
plt.plot(x2, fit21, color = 'black')

params = [-1, -1, 1, 1, 1, 1, 1]
bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (0, 0, np.inf, np.inf, np.inf, np.inf, np.inf))

fit_params22, fit_cov = sp.optimize.curve_fit(exp_poly_fit, xbin_centers[xbin_centers > 0.7], -s*sigma_fit[(xbin_centers > 0.7)], p0 = params, bounds = bounds)
fit22 = exp_poly_fit(x2, *fit_params22)
plt.plot(x2, fit22, color = 'black')

###Plotting without points
plt.figure()
dtime = plt.hist2d(np.array(p3_pip), np.array(dt_pip), bins = (np.arange(0, 6, 0.05), np.arange(-3, 3, 0.05)), range = ((0, 6), (-3, 3)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.plot(x2, fit21, color = 'red')
plt.plot(x2, fit22, color = 'red')

#%%

### Creating mask array to remove values from cut

mask21 = dt_p < exp_poly_fit(p3_p, *fit_params21)
mask22 = dt_p > exp_poly_fit(p3_p, *fit_params22)

tp_mask = mask21 & mask22

### Plotting cut 2d hist
plt.figure()
dtime = plt.hist2d(np.array(p3_p)[tp_mask], np.array(dt_p)[tp_mask], bins = (np.arange(0, 6, 0.05), np.arange(-2, 2, 0.05)), range = ((0, 6), (-2, 2)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.plot(x2, fit21, color = 'red')
plt.plot(x2, fit22, color = 'red')


#%% Plotting cut results
plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(tpb_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(tpb_mm[~tp_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(tpb_mm[tp_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

#%%
###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.6, 1.5 + 0.015, 0.015)
bin_content, bin_edges, _ = plt.hist((tpb_mm[tp_mask]), bins = bins, range = (0.6, 1.5))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(data.MM[wmask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

tp_mm = tpb_mm[tp_mask]


#%%
### Chi2 PID is aparameter that is calculated by JLab. It is a measure of how
### certain we are that a particle is actually the particle that we say it is.
### many paramters go into its calculation so i wont go into it. 

chi2_pb= np.array(data.chi2pid_pb)[wmask][mask][tp_mask] #whenever i load the variable it is uncut so i apply the cuts so array size will match

plt.figure()
plt.hist(chi2_pb, bins = 100, range = (-50, 50))
plt.title(r"$\chi^2$ PID for Antiproton", fontsize = 12.5)
plt.xlabel(r"$\chi^2$ PID", fontsize = 12.5)
plt.axvline(x = 8, color = 'red', label = r'Upper Bound = $8$')
plt.axvline(x = -6, color = 'red', label = r'Lower Bound = $-6$')
plt.legend()

chi2_pb_mask = (chi2_pb < 8) & (chi2_pb > -6) #These values were determined through trial and error.

#%% Plotting cut results
plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(tp_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(tp_mm[~chi2_pb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(tp_mm[chi2_pb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\chi^2$ PID Cut (eFD)")

#%%
###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(tp_mm[chi2_pb_mask], bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.title('Electron in FD')
plt.ylabel("Counts / 15 MeV")

chipb_mm = tp_mm[chi2_pb_mask]

#%%
#Same cut now on proton.
chi2_p= np.array(data.chi2pid_pb[wmask][mask][tp_mask][chi2_pb_mask])

plt.figure()
plt.hist(chi2_p, bins = 100, range = (-10, 10))

chi2_p_mask = (chi2_p < 6) & (chi2_p > -6)

#%% Plotting cut results
plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(chipb_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(chipb_mm[~chi2_p_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(chipb_mm[chi2_p_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

#%%
###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(chipb_mm[chi2_p_mask], bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

chip_mm = tp_mm[chi2_pb_mask]

#%%
#AGain but piplus
chi2_pip = np.array(data.chi2pid_pb[wmask][mask][tp_mask][chi2_pb_mask])

plt.figure()
plt.hist(chi2_pip, bins = 100, range = (-10, 10))

chi2_pip_mask = (chi2_pip < 6) & (chi2_pip > -6)

#%% Plotting cut results
plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(chipb_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(chipb_mm[~chi2_pip_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(chipb_mm[chi2_pip_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

#%%
###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(chipb_mm[chi2_p_mask], bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

chip_mm = tp_mm[chi2_pb_mask]

#%%

### I guess here i decided to redo the antiproton delta time cut??? not sure
### why i did it again in a seperate cell but dont be confused lol.
p3_pb, dt_pb = np.array(data['P_mag_pb'])[wmask][mask][tp_mask][chi2_pb_mask], np.array(data['deltaTime_pb'])[wmask][mask][tp_mask][chi2_pb_mask]

### 2d hist showing momentum dependence of delta time
plt.figure()
dtime = plt.hist2d(np.array(p3_pb), np.array(dt_pb), bins = (np.arange(0, 6, 0.05), np.arange(-2, 2, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\bar{p}$')


H, xedges, yedges, img = dtime

#%%

### Defining arrays for fitting parameters from 1D histograms
A_fit = np.empty(len(H[:,:]))
mu_fit = np.empty_like(A_fit)
sigma_fit = np.empty_like(A_fit)


bin_centers = ((yedges[:-1] + yedges[1:])/2)
#bin_centers = pre_bin_centers[(pre_bin_centers >= -1) & (pre_bin_centers <= 1)]

###Slicing 2D histogram into 1D dt slices for a given momentum, then fitting with gaussian
for i in range(len(H[:,:])): 
    sliced_hist = H[i, :]
    
    if np.sum(sliced_hist) == 0:
        continue
    
    params = [50000, 0, 0.986]
    bounds = ((0, -1, 0), (100000, 1, 5))
    
    try:
        fit_params, fit_cov = sp.optimize.curve_fit(gauss_fit, bin_centers, sliced_hist, p0 = params, bounds = bounds)
        A_fit[i], mu_fit[i], sigma_fit[i] = fit_params[:3]
    except RuntimeError:
        print(f"Gaussian fit failed for column {i}")
        continue  # Skip to next column if fit fails
    
    # plt.figure()
    # plt.bar(bin_centers, sliced_hist, width = yedges[1] - yedges[0])
    # x = np.linspace(-2, 2, 1000)
    # fit = gauss_fit(x, *fit_params)
    # plt.plot(x, fit, color = 'red')
    
        
### Plotting a number of sigma values from our fits on 2D hist

s = 2 ### # of sigma i am cutting at
xbin_centers = ((xedges[:-1] + xedges[1:])/2)
cut_sigma_fit = sigma_fit[xbin_centers > 0.2]
sigma_vals = np.append(s*cut_sigma_fit, -s*cut_sigma_fit)

plt.figure()
dtime = plt.hist2d(np.array(p3_pb), np.array(dt_pb), bins = (np.arange(0, 6, 0.05), np.arange(-1.5, 1.5, 0.05)), range = ((0, 6), (-1.5, 1.5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
plt.scatter(np.append(xbin_centers[xbin_centers > 0.2], xbin_centers[xbin_centers > 0.2]), sigma_vals, color = 'red', s = 10)

plt.figure()
plt.hist(dt_pb, bins = 100)

dtpb_mask = (dt_pb < 0.4) #& (dt_pb > -0.5)

plt.figure()
dtime = plt.hist2d(np.array(p3_pb)[dtpb_mask], np.array(dt_pb)[dtpb_mask], bins = (np.arange(0, 6, 0.05), np.arange(-5, 5, 0.05)), range = ((0, 6), (-5, 5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\bar{p}$')

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(chipb_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(chipb_mm[~dtpb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(chipb_mm[dtpb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

#%%
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.6, 1.5, 0.015)
bin_content, bin_edges, _ = plt.hist(chipb_mm[dtpb_mask], bins = bins, range = (0.6, 1.5))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.6, 1.5, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(tpb_mm[chi2_pb_mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3f} $\pm$ {A_uncertainty:.3f}\n$\mu$ = {fit_params[1]:.3f} $\pm$ {mu_uncertainty:.3f}\n$\sigma$ = {fit_params[2]:.3f} $\pm$ {sigma_uncertainty:.3f}\nYield = {fit_yield:.0f} $\pm$ {yield_uncertainty:.3f}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend()
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%

### Chi2 proton cut that was ineffective

# chi2_p= np.array(data.chi2pid_p[wmask][mask][chi2_pb_mask])

# plt.figure()
# plt.hist(chi2_p, bins = 100, range = (-10, 10))

# chi2_p_mask = (chi2_p < 7) & (chi2_p > -7)

# #%% Plotting cut results
# plt.figure()
# bins = np.arange(0, 3 + 0.015, 0.015)
# plt.hist(chipb_mm, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
# plt.hist(chipb_mm[~chi2_p_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
# plt.hist(chipb_mm[chi2_p_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
# plt.legend()
# plt.xlabel("Mass (GeV)")
# plt.title(rf"Results of $\Delta t$ cut at {s}$\sigma$")

# #%%
# ###Fitting and plotting results
# plt.figure()
# params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
# bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

# bins = np.arange(0.6, 1.5 + 0.015, 0.015)
# bin_content, bin_edges, _ = plt.hist((chipb_mm[chi2_p_mask]), bins = bins, range = (0.6, 1.5))

# bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

# fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

# x = np.linspace(0.6, 1.5, 10000)
# fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015

# stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

# plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
# plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
# plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
# plt.axvline(x = 1, color = 'none', label = f"Mean = {fit_params[1]:.3f}\nSigma = {fit_params[2]:.3f}\nYield = {fit_yield:.0f}")
# plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
# plt.legend()

#%%
###The next section contains particle counter cuts. We are looking at a very specifric
### final state so we can choose to remove events that contain a different number of 
### particles type or charge tghan what we are expecting. Of course, only if it improves
### the peak to background ratio.

particle_dict = {
    11: "e",
    111: "pi0",
    211: "pip",
    -211: "pim",
    2212: 'p',
    2112: 'n',
    -2212: 'pb',
    -2112: 'nb'
}

particleID = data['ParticleID'][wmask][mask][tp_mask][chi2_pb_mask]#[tpb_mask]

ak_pid = ak.Array(particleID) #this was done using awkward arrays due to their vectorized way of dealing with arrays
#it is much faster.

#%%

### Pi-Plus particle counter cut

particle = "pip"

code = [code for code, name in particle_dict.items() if name == particle][0]

pip_counts = ak.sum(ak_pid == code, axis = 1)

plt.figure()
plt.hist2d(np.array(pip_counts), np.array(chipb_mm), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs $\pi^+$ Particle Counter')
plt.xlabel(r"$\pi^+$ Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")
#%%
pip_mask = pip_counts <= 1

mm_2 = chipb_mm

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_2, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_2[~pip_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_2[pip_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\pi^+$ Particle Counter Cut")

mm_pip = mm_2[pip_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_pip, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%
### Proton particle counter cut

particle = "p"

code = [code for code, name in particle_dict.items() if name == particle][0]

p_counts = ak.sum(ak_pid == code, axis = 1)

plt.figure()
plt.hist2d(np.array(p_counts[pip_mask]), np.array(mm_pip), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs $p$ Particle Counter')
plt.xlabel(r"$p$ Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

p_mask = p_counts[pip_mask] <= 1

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pip, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pip[~p_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pip[p_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $p$ Particle Counter Cut")
#%%
mm_p = mm_pip[p_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_p, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%
### AntiProton particle counter cut

particle = "pb"

code = [code for code, name in particle_dict.items() if name == particle][0]

pb_counts = ak.sum(ak_pid == code, axis = 1)

plt.figure()
plt.hist2d(np.array(pb_counts[pip_mask]), np.array(mm_pip), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs $\bar{p}$ Particle Counter')
plt.xlabel(r"$\bar{p}$ Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

pb_mask = pb_counts[pip_mask] <= 1

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pip, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pip[~pb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pip[pb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\bar{p}$ Particle Counter Cut")
plt.xlabel("Mass (GeV)")
plt.title(r'MM($ep\bar{p}\pi^+$)')
#%%
mm_pb = mm_pip[pb_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_pb, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%
### non-pip pion particle counter cut

particle1 = "pim"
particle2 = 'pi0'

code1 = [code for code, name in particle_dict.items() if name == particle1][0]
code2 = [code for code, name in particle_dict.items() if name == particle2][0]

pi_counts = ak.sum((ak_pid == code1) |(ak_pid == code2) , axis = 1)

plt.figure()
plt.hist2d(np.array(pi_counts[pip_mask]), np.array(mm_pip), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs ($\pi^0/\pi^-$) Particle Counter')
plt.xlabel(r"($\pi^0/\pi^-$) Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

pi_mask = pi_counts[pip_mask] < 1

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pip, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pip[~pi_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pip[pi_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of ($\pi^0/\pi^-$) Particle Counter Cut")
#%%
mm_pi = mm_pip[pi_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_pi, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%

### non-pip pion particle counter cut

particle = "e"

code = [code for code, name in particle_dict.items() if name == particle][0]

e_counts = ak.sum((ak_pid == code), axis = 1)

plt.figure()
plt.hist2d(np.array(e_counts[pip_mask][pi_mask]), np.array(mm_pi), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs $e^-$ Particle Counter')
plt.xlabel(r"$e^-$ Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

e_mask = e_counts[pip_mask][pi_mask] <= 2

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pi, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pi[~e_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pi[e_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $e^-$ Particle Counter Cut")

#%%
mm_e = mm_pi[e_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_e, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(mm_e)}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3f} $\pm$ {A_uncertainty:.3f}\n$\mu$ = {fit_params[1]:.3f} $\pm$ {mu_uncertainty:.3f}\n$\sigma$ = {fit_params[2]:.3f} $\pm$ {sigma_uncertainty:.3f}\nYield = {fit_yield:.0f} $\pm$ {yield_uncertainty:.3f}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend()
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%
### Now we do the cuts based on charge

# Dictionary for positively charged particles
positive_particles = {
    -11: "positron (e+)",
    13: "mu+",
    15: "tau+",
    211: "pi+ (pip)",
    321: "K+",
    2212: "proton (p)",
    4122: "Lambda_c+",
    423: "D+*",
    411: "D+",
    431: "Ds+",
    521: "B+",
    531: "Bs+",
    541: "Bc+"
}

# Dictionary for negatively charged particles
negative_particles = {
    11: "electron (e-)",
    -13: "mu-",
    -15: "tau-",
    -211: "pi- (pim)",
    -321: "K-",
    -2212: "antiproton (pbar)",
    -4122: "anti-Lambda_c-",
    -423: "D-*",
    -411: "D-",
    -431: "Ds-",
    -521: "B-",
    -531: "Bs-",
    -541: "Bc-"
}

# Dictionary for neutral particles
neutral_particles = {
    12: "neutrino (νe)",
    14: "muon neutrino (νμ)",
    16: "tau neutrino (ντ)",
    22: "photon (γ)",
    111: "pi0",
    130: "K0L",
    310: "K0S",
    311: "K0",
    3212: "Sigma0",
    #2112: "neutron (n)",
    421: "D0",
    511: "B0",
    531: "Bs0"
}

particleID = data['ParticleID'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask]

ak_pid = ak.Array(particleID)

#%%
# Get the PDG codes of all positive particles
positive_codes = list(positive_particles.keys())

ak_pid_flattened = ak.flatten(ak_pid, axis=-1)

# Create a mask for positive particles in the ak_pid array
# Compare each particle in ak_pid_flattened with the positive_codes
positive_mask = ak.any(ak_pid_flattened[:, None] == ak.Array(positive_codes)[None, :], axis=-1)

# Reshape positive_mask to match the original event structure
positive_mask_reshaped = ak.unflatten(positive_mask, ak.num(ak_pid, axis=1))

# Count the number of positive particles in each event
pos_counts = ak.sum(positive_mask_reshaped, axis=1)

plt.figure()
plt.hist2d(np.array(pos_counts), np.array(mm_pi), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs $\pi^+$ Particle Counter')
plt.xlabel(r"$\pi^+$ Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

#%%

pos_mask = pos_counts <= 2

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pi, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pi[~pos_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pi[pos_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\pi^+$ Particle Counter Cut")

mm_pos = mm_pi[pos_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_pos, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")


#%%
# Get the PDG codes of all negative particles
negative_codes = list(negative_particles.keys())

ak_pid_flattened = ak.flatten(ak_pid, axis=-1)

# Create a mask for negative particles in the ak_pid array
# Compare each particle in ak_pid_flattened with the negative_codes
negative_mask = ak.any(ak_pid_flattened[:, None] == ak.Array(negative_codes)[None, :], axis=-1)

# Reshape negative_mask to match the original event structure
negative_mask_reshaped = ak.unflatten(negative_mask, ak.num(ak_pid, axis=1))

# Count the number of negative particles in each event
neg_counts = ak.sum(negative_mask_reshaped, axis=1)

plt.figure()
plt.hist2d(np.array(neg_counts[pos_mask]), np.array(mm_pos), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs $\pi^+$ Particle Counter')
plt.xlabel(r"$\pi^+$ Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

#%%

neg_mask = neg_counts[pos_mask] <= 2

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pos, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pos[~neg_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pos[neg_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\pi^+$ Particle Counter Cut")

mm_neg = mm_pos[neg_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_neg, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")


#%%
# Get the PDG codes of all neutral particles
neutral_codes = list(neutral_particles.keys())

ak_pid_flattened = ak.flatten(ak_pid, axis=-1)

# Create a mask for neutral particles in the ak_pid array
# Compare each particle in ak_pid_flattened with the neutral_codes
neutral_mask = ak.any(ak_pid_flattened[:, None] == ak.Array(neutral_codes)[None, :], axis=-1)

# Reshape neutral_mask to match the original event structure
neutral_mask_reshaped = ak.unflatten(neutral_mask, ak.num(ak_pid, axis=1))

# Count the number of neutral particles in each event
neu_counts = ak.sum(neutral_mask_reshaped, axis=1)

plt.figure()
plt.hist2d(np.array(neu_counts[pos_mask]), np.array(mm_pos), norm = mpl.colors.LogNorm(), bins = (np.arange(0, max(pip_counts)+2), (np.arange(0, 3 + 0.015, 0.015))))
plt.title(r'MM($ep\bar{p}\pi^+$) vs Neutral Particle Counter')
plt.xlabel(r"Neutral Particle Counter")
plt.ylabel(r"MM($ep\bar{p}\pi^+$)")

#%%
neu_mask = neu_counts[pos_mask] <= 7

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_pos, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_pos[~neu_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_pos[neu_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\pi^+$ Particle Counter Cut")

mm_neu = mm_pos[neu_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_neu, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%
### TOF mass cuts we are calclating the mass of the oparticle using various parameters and comparing
### it to the known mass of the particle. We then cut on this at whatever value it is necessary.
### See thesis for definition.

#TOF Mass for pip

beta_pip = data['beta_pip'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask]
p3_pip = data['P_mag_pip'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask]

tofm_pip = (p3_pip/beta_pip)*np.sqrt(np.abs(1-beta_pip**2))

plt.figure()
tofpip = plt.hist2d(np.array(p3_pip), np.array(tofm_pip), bins = (np.arange(0, 6, 0.05), np.arange(0, 0.4, 0.015)), range = ((0, 6), (0, 0.4)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('TOF Mass')

bins = (np.arange(-1, 1, 0.010))
plt.figure()
plt.hist(tofm_pip, bins = bins)

H, xedges, yedges, img = tofpip

#%%

### Defining arrays for fitting parameters from 1D histograms
A_fit = np.empty(len(H[:,:]))
mu_fit = np.empty_like(A_fit)
sigma_fit = np.empty_like(A_fit)


bin_centers = ((yedges[:-1] + yedges[1:])/2)
#bin_centers = pre_bin_centers[(pre_bin_centers >= -1) & (pre_bin_centers <= 1)]

###Slicing 2D histogram into 1D dt slices for a given momentum, then fitting with gaussian
for i in range(len(H[:,:])): 
    sliced_hist = H[i, :]
    
    if np.sum(sliced_hist) == 0:
        continue
    
    params = [50000, 0, 0.986]
    bounds = ((0, -1, 0), (100000, 1, 5))
    
    try:
        fit_params, fit_cov = sp.optimize.curve_fit(gauss_fit, bin_centers, sliced_hist, p0 = params, bounds = bounds)
        A_fit[i], mu_fit[i], sigma_fit[i] = fit_params[:3]
    except RuntimeError:
        print(f"Gaussian fit failed for column {i}")
        continue  # Skip to next column if fit fails
    
    # plt.figure()
    # plt.bar(bin_centers, sliced_hist, width = yedges[1] - yedges[0])
    # x = np.linspace(-2, 2, 1000)
    # fit = gauss_fit(x, *fit_params)
    # plt.plot(x, fit, color = 'red')
    
        
### Plotting a number of sigma values from our fits on 2D hist

s = 2 ### # of sigma i am cutting at

# mask = np.ones(len(xbin_centers), dtype=bool)  # Start with all True
# mask[[105, 108]] = False    
xbin_centers = ((xedges[:-1] + xedges[1:])/2)
sigma_vals = np.append((s*sigma_fit + mu_fit), (-s*sigma_fit + mu_fit))

plt.figure()
tofpip = plt.hist2d(np.array(p3_pip), np.array(tofm_pip), bins = (np.arange(0, 6, 0.05), np.arange(0, 0.4, 0.015)), range = ((0, 6), (0, 0.4)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('TOF Mass')
plt.scatter(np.append(xbin_centers, xbin_centers), sigma_vals, color = 'red', s = 10)


#%%
tofpip_mask = (tofm_pip < 0.535) & (tofm_pip > 0)

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_neu, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_neu[~tofpip_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_neu[tofpip_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\pi^+$ TOF Mass Cut")

mm_tofpip = mm_neu[tofpip_mask]

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_tofpip, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Mass $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")

#%%
#TOF Mass for pbar

beta_pb = data['beta_pb'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask]
p3_pb = data['P_mag_pb'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask]

tofm_pb = (p3_pb/beta_pb)*np.sqrt(1-beta_pb**2)

plt.figure()
tofpb = plt.hist2d(np.array(p3_pb), np.array(tofm_pb), bins = (np.arange(0, 6, 0.05), np.arange(0.5, 1.5, 0.015)), range = ((0, 6), (0.5, 1.5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('TOF Mass')
plt.title('TOF Mass vs Momentum for Antiproton')

bins = (np.arange(0.5, 1.5, 0.010))
plt.figure()
plt.hist(tofm_pb, bins = bins)
plt.ylabel(r'TOF Mass $[GeV/c^2]$')
plt.title(r'TOF Mass for $\bar{p}$')
plt.axvline(x = 1.1, color = 'red')
plt.axvline(x = 0.83, color = 'red')
H, xedges, yedges, img = tofpb

#%%

### Defining arrays for fitting parameters from 1D histograms
A_fit = np.empty(len(H[:,:]))
mu_fit = np.empty_like(A_fit)
sigma_fit = np.empty_like(A_fit)


bin_centers = ((yedges[:-1] + yedges[1:])/2)
#bin_centers = pre_bin_centers[(pre_bin_centers >= -1) & (pre_bin_centers <= 1)]

###Slicing 2D histogram into 1D dt slices for a given momentum, then fitting with gaussian
for i in range(len(H[:,:])): 
    sliced_hist = H[i, :]
    
    if np.sum(sliced_hist) == 0:
        continue
    
    params = [50000, 0, 0.986]
    bounds = ((0, -1, 0), (100000, 1, 5))
    
    try:
        fit_params, fit_cov = sp.optimize.curve_fit(gauss_fit, bin_centers, sliced_hist, p0 = params, bounds = bounds)
        A_fit[i], mu_fit[i], sigma_fit[i] = fit_params[:3]
    except RuntimeError:
        print(f"Gaussian fit failed for column {i}")
        continue  # Skip to next column if fit fails
    
    # plt.figure()
    # plt.bar(bin_centers, sliced_hist, width = yedges[1] - yedges[0])
    # x = np.linspace(-2, 2, 1000)
    # fit = gauss_fit(x, *fit_params)
    # plt.plot(x, fit, color = 'red')
    
        
### Plotting a number of sigma values from our fits on 2D hist

s = 5 ### # of sigma i am cutting at

# mask = np.ones(len(xbin_centers), dtype=bool)  # Start with all True
# mask[[105, 108]] = False    
xbin_centers = ((xedges[:-1] + xedges[1:])/2)
sigma_vals = np.append((s*sigma_fit + mu_fit), (-s*sigma_fit + mu_fit))

plt.figure()
tofpb = plt.hist2d(np.array(p3_pb), np.array(tofm_pb), bins = (np.arange(0, 6, 0.05), np.arange(0.5, 1.5, 0.015)), range = ((0, 6), (0.5, 1.5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('TOF Mass')
plt.scatter(np.append(xbin_centers, xbin_centers), sigma_vals, color = 'red', s = 10)

params = [0.005, 1.5, 1, 1, 1, 1, 1]
bounds = ((0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (10, 10, np.inf, np.inf, np.inf, np.inf, np.inf))

x2 = np.linspace(0, 6, 10000)

fit_params21, fit_cov = sp.optimize.curve_fit(exp_poly_fit, xbin_centers[(xbin_centers > 1) & (xbin_centers < 4)], ((s*sigma_fit + mu_fit)[(xbin_centers > 1) & (xbin_centers < 4)]), p0 = params, bounds = bounds)
fit21 = exp_poly_fit(x2, *fit_params21)
plt.plot(x2, fit21, color = 'red')
params = [-0.005, 1.5, 1, 1, 1, 1, 1]
bounds = ((-10, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (0, 10, np.inf, np.inf, np.inf, np.inf, np.inf))

fit_params22, fit_cov = sp.optimize.curve_fit(exp_poly_fit, xbin_centers[(xbin_centers > 1) & (xbin_centers < 4)], ((-s*sigma_fit + mu_fit)[(xbin_centers > 1) & (xbin_centers < 4)]), p0 = params, bounds = bounds)
fit22 = exp_poly_fit(x2, *fit_params22)
plt.plot(x2, fit22, color = 'red')


#%%

### Creating mask array to remove values from cut

# mask11 = dt_pip[p3_pip < 0.6] < poly_fit(p3_pip[p3_pip < 0.6], *fit_params11)
# mask12 = dt_pip[p3_pip < 0.6] > poly_fit(p3_pip[p3_pip < 0.6], *fit_params12)
mask21 = tofm_pb < exp_poly_fit(p3_pb, *fit_params21)
mask22 = tofm_pb > exp_poly_fit(p3_pb, *fit_params22)

mask2 = np.zeros_like(dt_pip, dtype=bool)

# mask[p3_pip < 0.6] = mask11 & mask12
mask2 = mask21 & mask22

### Plotting cut 2d hist
plt.figure()
tofpb = plt.hist2d(np.array(p3_pb)[mask2], np.array(tofm_pb)[mask2], bins = (np.arange(0, 6, 0.05), np.arange(0.5, 1.5, 0.015)), range = ((0, 6), (0.5, 1.5)), norm = mpl.colors.LogNorm())
plt.xlabel('Momentum Magnitude (GeV)')
plt.ylabel('Delta Time (ns)')
plt.title(r'$\Delta t$ vs Momentum for $\pi ^+$')
# plt.plot(x1, fit11, color = 'red')
# plt.plot(x1, fit12, color = 'red')
plt.plot(x2, fit21, color = 'red')
plt.plot(x2, fit22, color = 'red')


#%%
tofpb_mask = (((tofm_pb < 1.1) & (tofm_pb > 0.78)))

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mm_tofpip, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(mm_tofpip[~(tofpb_mask)], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(mm_tofpip[tofpb_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"Results of $\bar{p}$ TOF Mass Cut")

mm_tofpb = mm_tofpip[tofpb_mask]

#%%

###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_tofpb, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(mm_tofpb)}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.0f} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'Missing Mass $[GeV/c^2]$', fontsize = 12.5)
plt.ylabel("Counts / 15 MeV", fontsize = 12.5)
plt.title(r"Missing Mass Distribution with Cuts", fontsize = 17.5)

# #%%
# #TOF Mass for p

# beta_p = data['beta_p'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]
# p3_p = data['P_mag_p'][wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]

# tofm_p = (p3_p/beta_p)*np.sqrt(1-beta_p**2)

# plt.figure()
# tofpb = plt.hist2d(np.array(p3_p), np.array(tofm_p), bins = (np.arange(0, 6, 0.05), np.arange(0.5, 1.5, 0.015)), range = ((0, 6), (0.5, 1.5)), norm = mpl.colors.LogNorm())
# plt.xlabel('Momentum Magnitude (GeV)')
# plt.ylabel('TOF Mass')

# bins = (np.arange(0.5, 1.5, 0.010))
# plt.figure()
# plt.hist(tofm_p, bins = bins)

# H, xedges, yedges, img = tofpb

# #%%

# tofp_mask = (tofm_p < 1) & (tofm_p > 0.9)

# plt.figure()
# bins = np.arange(0, 3 + 0.015, 0.015)
# plt.hist(mm_tofpb, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
# plt.hist(mm_tofpb[~(tofp_mask)], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
# plt.hist(mm_tofpb[tofp_mask], bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
# plt.legend()
# plt.xlabel("Mass (GeV)")
# plt.title(r"Results of $p$ TOF Mass Cut")

# mm_tofp = mm_tofpb[tofp_mask]

# ###Fitting and plotting results
# plt.figure()
# params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
# bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

# bins = np.arange(0.55, 1.45, 0.015)
# bin_content, bin_edges, _ = plt.hist(mm_tofp, bins = bins, range = (0.55, 1.45))

# bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

# fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

# x = np.linspace(0.55, 1.45, 10000)
# fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
# plt.close()
# plt.figure()
# A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
# yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

# plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
# plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

# stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

# db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
# dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

# plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
# plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
# plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
# plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
# plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
# plt.legend(fontsize = 9)
# plt.xlabel(r'$MM(ep\bar{p}\pi^+$) $[GeV/c^2]$')
# plt.ylabel("Counts / 15 MeV")

#%%
###Fitting and plotting results
plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.55, 1.45, 0.015)
bin_content, bin_edges, _ = plt.hist(mm_tofpb, bins = bins, range = (0.55, 1.45))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.55, 1.45, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = fit_params[1] - 3*fit_params[2], color = 'green', label = r'$3\sigma$')
plt.axvline(x = fit_params[1] - 6*fit_params[2], color = 'blue', label = r'$6\sigma$')
plt.axvline(x = fit_params[1] + 3*fit_params[2], color = 'green')
plt.axvline(x = fit_params[1] + 6*fit_params[2], color = 'blue')

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red')
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--')
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan')
plt.axvline(x = 1, color = 'none')
plt.axvline(x = 1, color = 'none')
#plt.legend(fontsize = 9)
plt.xlabel(r'$MM(ep\bar{p}\pi^+$) $[GeV/c^2]$')
plt.ylabel("Counts / 15 MeV")
plt.legend()

### This next opart is not super important its sort of a prelimanry background subtraction.
### You can estimate the background under the peak by looking at the background right next to
### the peak on either side.

def integrand(x):
    return gauss_poly_fit(x, *fit_params)

def integrand2(x):
    return poly_fit(x, *fit_params[3:])

sb1 = sp.integrate.quad(integrand, (fit_params[1] - 6*fit_params[2]), (fit_params[1] - 3*fit_params[2]))[0]/0.015
sb2 = sp.integrate.quad(integrand, (fit_params[1] + 3*fit_params[2]), (fit_params[1] + 6*fit_params[2]))[0]/0.015
sb = sb1+sb2

peak_bkg = sp.integrate.quad(integrand2, (fit_params[1] - 3*fit_params[2]), (fit_params[1] + 3*fit_params[2]))[0]/0.015

scaling = (peak_bkg/sb)

peak_rgn = (mm_tofpb <= fit_params[1] + 3*fit_params[2]) & (mm_tofpb >= fit_params[1] - 3*fit_params[2])

sb_rgn = ((mm_tofpb >= fit_params[1] - 6*fit_params[2]) & (mm_tofpb <= fit_params[1] - 3*fit_params[2])) | ((mm_tofpb >= fit_params[1] + 3*fit_params[2]) & (mm_tofpb <= fit_params[1] + 6*fit_params[2]))
#%%
mass_n = 0.9395654133
p_target = vec.obj(px = 0, py = 0, pz = 0, E = mass_p)
p_beam = vec.obj(px = 0, py = 0, pz = 10.2, E = 10.2)
cut_p_e = vec.array({'px': px_e, 'py': py_e, 'pz': pz_e, 'M': np.ones_like(pz_e)*mass_e})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_p = vec.array({'px': px_p, 'py': py_p, 'pz': pz_p, 'M': np.ones_like(pz_e)*mass_p})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_pb = vec.array({'px': px_pb, 'py': py_pb, 'pz': pz_pb, 'M': np.ones_like(pz_e)*mass_p})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_pip = vec.array({'px': px_pip, 'py': py_pip, 'pz': pz_pip, 'M': np.ones_like(pz_e)*mass_pip})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_miss = p_beam + p_target - cut_p_e - cut_p_p - cut_p_pb - cut_p_pip
cut_p_n = vec.array({'px': cut_p_miss.px[peak_rgn], 'py': cut_p_miss.py[peak_rgn], 'pz': cut_p_miss.pz[peak_rgn], 'M': np.ones_like(cut_p_miss.pz[peak_rgn])*mass_n})
cut_p_sb = vec.array({'px': cut_p_miss.px[sb_rgn], 'py': cut_p_miss.py[sb_rgn], 'pz': cut_p_miss.pz[sb_rgn], 'M': np.ones_like(cut_p_miss.pz[sb_rgn])*mass_n})

res = (cut_p_n + cut_p_pb[peak_rgn]).M

sb_res = ((cut_p_sb + cut_p_pb[sb_rgn]).M)

bins = np.arange(1.5, 3, 0.020)
bin_content1, bin_edges, _ = plt.hist(res, bins = bins, range = (1.5, 3))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

bin_content2, bin_edges, _ = plt.hist(sb_res, bins = bins, range = (1.5, 3))

bin_content3 = bin_content1 - scaling*bin_content2

plt.figure()
plt.bar(bin_centers, bin_content3, width = 0.020)
plt.errorbar(bin_centers, bin_content3, yerr=np.sqrt(bin_content1 + scaling*bin_content2), fmt = 'none', color = 'black')
plt.title('Mass of Antiproton-Neutron System')
plt.xlabel(r'M($\bar{p}n$) $[GeV/c^2]$', fontsize = 12.5)
plt.ylabel('Counts / 20 MeV', fontsize = 12.5)
plt.title(r'Mass of $\bar{p}n$ System for Electron in FD', fontsize = 17.5)

#%%

res = (cut_p_n + cut_p_p[peak_rgn]).M

sb_res = ((cut_p_sb + cut_p_p[sb_rgn]).M)

bins = np.arange(1.5, 3, 0.020)
bin_content1, bin_edges, _ = plt.hist(res, bins = bins, range = (1.5, 3))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

bin_content2, bin_edges, _ = plt.hist(sb_res, bins = bins, range = (1.5, 3))

bin_content3 = bin_content1 - scaling*bin_content2

plt.figure()
plt.bar(bin_centers, bin_content3, width = 0.020)
plt.errorbar(bin_centers, bin_content3, yerr=np.sqrt(bin_content1 + scaling*bin_content2), fmt = 'none', color = 'black')
plt.title('Mass of Antiproton-Neutron System')
plt.xlabel(r'M($pn$) $[GeV/c^2]$', fontsize = 12.5)
plt.ylabel('Counts / 20 MeV', fontsize = 12.5)
plt.title(r'Mass of $pn$ System for Electron in FD', fontsize = 17.5)

#%%



#%%

p_target = vec.obj(px = 0, py = 0, pz = 0, E = mass_p)
p_beam = vec.obj(px = 0, py = 0, pz = 10.2, E = 10.2)
cut_p_e = vec.array({'px': px_e, 'py': py_e, 'pz': pz_e, 'M': np.ones_like(pz_e)*mass_e})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_p = vec.array({'px': px_p, 'py': py_p, 'pz': pz_p, 'M': np.ones_like(pz_e)*mass_p})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_pb = vec.array({'px': px_pb, 'py': py_pb, 'pz': pz_pb, 'M': np.ones_like(pz_e)*mass_p})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]
cut_p_pip = vec.array({'px': px_pip, 'py': py_pip, 'pz': pz_pip, 'M': np.ones_like(pz_e)*mass_pip})[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]#[(mm_tofpb > 0.82) & (mm_tofpb < 1.1)]

#%%
cut_px_pip = np.empty(len(cut_p_pip))
cut_py_pip = np.empty(len(cut_p_pip))
cut_pz_pip = np.empty(len(cut_p_pip))
cut_E_pip = np.empty(len(cut_p_pip))

for i in range(len(cut_p_pip) - 1):
    cut_px_pip[i] = cut_p_pip.px[i+1]
    cut_py_pip[i] = cut_p_pip.py[i+1]
    cut_pz_pip[i] = cut_p_pip.pz[i+1]
    cut_E_pip[i] = cut_p_pip.E[i+1]
    
cut_px_pip[len(cut_p_pip) - 1] = cut_p_pip.px[0]
cut_py_pip[len(cut_p_pip) - 1] = cut_p_pip.py[0]
cut_pz_pip[len(cut_p_pip) - 1] = cut_p_pip.pz[0]
cut_E_pip[len(cut_p_pip) - 1] = cut_p_pip.E[0]

cut_p_pip = vec.array({'px': cut_px_pip, 'py': cut_py_pip, 'pz': cut_pz_pip, 'M': cut_E_pip})



#%%
### Defining MM(p pb pip e)
cut_p_miss = p_beam + p_target - cut_p_p - cut_p_pb - cut_p_pip - cut_p_e
cut_mass_miss = cut_p_miss.M

### Histogram of uncut missing mass
plt.figure()
bins = np.arange(0, 10, 0.015)
bin_content, bin_edges, _ = plt.hist(cut_mass_miss, bins = bins, range = (0.55, 1.45))
plt.axvline(x=mass_n, color='red', linestyle='--', linewidth=2, label = "Neutron mass: 939.6 MeV")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(r"MM($ep\bar{p}\pi^+$)")
plt.ylabel("Counts / 15MeV")

bin_content, bin_edges, _ = plt.hist(mm_tofpb, bins = bins, range = (0.55, 1.45))
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

plt.close()

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')

bin_content, bin_edges, _ = plt.hist(cut_mass_miss, bins = bins, range = (0.55, 1.45))



#%%

###Results of dt cuts

plt.figure()
params = [4000, mass_n, 0.02, 1, 1, 1, 1, 1]
bounds = ((0, 0.9, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,), (10000, 1.05, 1, np.inf, np.inf, np.inf, np.inf, np.inf,))

bins = np.arange(0.75, 1.2, 0.015)
bin_content, bin_edges, _ = plt.hist(np.append(tp_mm, mass_miss[~wmask]), bins = bins, range = (0.75, 1.2))

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

fit_params, fit_cov = sp.optimize.curve_fit(gauss_poly_fit, bin_centers, bin_content, p0 = params, bounds = bounds, sigma = np.sqrt(bin_content))

x = np.linspace(0.75, 1.2, 10000)
fit_yield = (np.sqrt(2*np.pi) * fit_params[0] * fit_params[2])/0.015
plt.close()
plt.figure()
A_uncertainty, mu_uncertainty, sigma_uncertainty = np.sqrt(np.diag(fit_cov))[:3]
yield_uncertainty = np.sqrt(((((np.sqrt(2*np.pi)*fit_params[2])/0.015)*A_uncertainty)**2) + ((((np.sqrt(2*np.pi)*fit_params[0])/0.015)*sigma_uncertainty)**2))

plt.errorbar(bin_centers, bin_content, yerr=np.sqrt(bin_content), fmt = 'none', color = 'black')
plt.axvline(x = 1, color = 'none', label = f"Events: {len(wcut_mm[mask])}")

stat_sig = fit_params[0]/(np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])))

db = np.sqrt(np.sqrt(np.diag(fit_cov))[3]**2 + (fit_params[1]*np.sqrt(np.diag(fit_cov))[4])**2 + ((fit_params[1]**2)*np.sqrt(np.diag(fit_cov))[5])**2 + ((fit_params[1]**3)*np.sqrt(np.diag(fit_cov))[6])**2 + ((fit_params[1]**4)*np.sqrt(np.diag(fit_cov))[7])**2) 
dsig = np.sqrt((((1/np.sqrt(fit_params[0] + poly_fit(fit_params[1], *fit_params[3:])) - (0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*A_uncertainty)**2) + (((0.5*fit_params[0])/((fit_params[0] + poly_fit(fit_params[1], *fit_params[3:]))**(3/2)))*db)**2)

plt.plot(x, gauss_poly_fit(x, *fit_params), color = 'red', label = "Background + Signal")
plt.plot(x, poly_fit(x, *fit_params[3:]), linestyle = '--', label = "Background")
plt.plot(x, gauss_fit(x, *fit_params[:3]), color = 'cyan', label = "Signal")
plt.axvline(x = 1, color = 'none', label = f"$A$ = {fit_params[0]:.3g} $\pm$ {A_uncertainty:.0f}\n$\mu$ = {fit_params[1]:.3g} $\pm$ {mu_uncertainty:.1g}\n$\sigma$ = {fit_params[2]:.3g} $\pm$ {sigma_uncertainty:.1g}\nYield = {fit_yield:.2g} $\pm$ {yield_uncertainty:.3g}")
plt.axvline(x = 1, color = 'none', label = f"Statistical\nSignificance = {stat_sig:.2f}")
plt.legend(fontsize = 9)
plt.xlabel(r'$m_{miss}$ $[GeV/c^2]$', fontsize = 12.5)
plt.ylabel("Counts / 15 MeV", fontsize = 12.5)
plt.title(r"Electron in FD", fontsize = 17.5)

plt.figure()
bins = np.arange(0, 3 + 0.015, 0.015)
plt.hist(mass_miss, bins = bins, range = (0, 3), alpha = 0.3, label = "No Cuts")
plt.hist(np.append(wcut_mm[~mask], dtpip_mm[~tp_mask]), bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Anti-Cut")
plt.hist(np.append(tp_mm, mass_miss[~wmask]), bins = bins, range = (0, 3), alpha = 0.3, label = r"$\Delta t$ Cut")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.title(rf"Results of $\Delta t$ cuts (eFD)")

#%%

arrays = tree.arrays(library="np")
#%%
### Here i am saving the final cut MM arrays into a new root file se we can do the background modeling 
### in a seperate file.
final_arrays = {}
for name, array in arrays.items():
    final_arrays[name] = array[wmask][mask][tp_mask][chi2_pb_mask][pip_mask][pi_mask][pos_mask][neu_mask][tofpip_mask][tofpb_mask]

with uproot.recreate("Cut_FD.root") as f:
    f["T"] = final_arrays



