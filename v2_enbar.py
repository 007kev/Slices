#%% PERPLEXITY CODE
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

f = uproot.open("v2_kev_Pppim_eFT_006667.root")
t = f["Individual"]

mm    = t["miss_mass"].array()
Enbar = t["Enbar_calo"].array()
ang   = t["neutral_angle"].array()
hasN  = t["has_neutral"].array()

mm    = ak.to_numpy(mm)
Enbar = ak.to_numpy(Enbar)
ang   = ak.to_numpy(ang)
hasN  = ak.to_numpy(hasN)

# 1) Apply neutron-like MM window
mm_low, mm_high = 0.8, 1.15
mm_window = (mm > mm_low) & (mm < mm_high)

# 2) Keep events with a neutral candidate and valid angle
neutral_ok = (hasN == 1) & (Enbar > 0.0) & (ang > 0.0)
sel_angle = mm_window & neutral_ok

ang_sel = np.rad2deg(ang[sel_angle])

print(f"Total events: {t.num_entries}")
print("Events in MM window:", np.count_nonzero(mm_window))
print("Events with neutral + valid angle:", len(ang_sel))


# 3) Histogram of neutral_angle
plt.figure()
plt.hist(ang_sel, bins=60, range=(0.0, 180), histtype="step")
plt.xlabel("neutral_angle (deg)")
plt.ylabel("Counts")
plt.title("Angle between neutral ECAL hit and missing momentum")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 4) After inspecting this plot, pick a theta_max
theta_max = 30 # adjust after looking at the distribution
angle_cut = (np.rad2deg(ang) > 0.0) & (np.rad2deg(ang) < theta_max)

sel = mm_window & neutral_ok & angle_cut
sampling_fraction = 0.12
Enbar_sel = Enbar[sel]/sampling_fraction

print(f"Events after {theta_max} angle cut:", np.count_nonzero(sel))

plt.figure()
plt.hist(Enbar_sel, bins=60, range=(0.0, 5.0), histtype="step")
plt.xlabel("Enbar_calo (GeV)")
plt.ylabel("Counts")
plt.title(f"Antineutron candidate energy (neutral_angle < {theta_max} deg)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Assuming sampling fraction is {sampling_fraction}')







# %% GEMINI CODE
import uproot
import matplotlib.pyplot as plt
import numpy as np

# 1. Open the file and extract the data
file = uproot.open("v2_kev_Pppim_eFT_006667.root")
tree = file["Individual"]
df = tree.arrays(["miss_mass", "neutral_angle", "e_status", "Enbar_calo"], library="pd")

# 2. Convert angle from radians to degrees
df['angle_deg'] = df['neutral_angle'] * (180.0 / np.pi)

# 3. Create a 2D Histogram (Missing Mass vs. Neutral Angle)
plt.figure(figsize=(10, 6))

# We look at Missing Mass around the neutron mass (0.5 to 1.5 GeV)
# and Angles from 0 to 50 degrees
plt.hist2d(df['miss_mass'], df['angle_deg'], 
           bins=[60, 60], range=[[0.9, 1.0], [0, 180]], 
           cmap='viridis', cmin=1)

plt.colorbar(label='Counts')
plt.title(r'Missing Mass of $e p \to e p p \pi^- X$ vs. ECAL Cluster Angle')
plt.xlabel('Missing Mass (GeV/$c^2$)')
plt.ylabel('Angle between $P_{miss}$ and Cluster (Degrees)')

# Draw lines where we expect the antineutron signal
plt.axvline(0.939, color='red', linestyle='--', alpha=0.7, label='Antineutron Mass (0.939 GeV)')
# plt.axhline(10, color='orange', linestyle='--', alpha=0.7, label='Typical 10° Cut')
plt.legend()
plt.tight_layout()
plt.show()


# Define the "Golden" criteria
# Angle < 10 degrees and Missing Mass within 50 MeV of the neutron mass
golden_cut = (df['angle_deg'] < 30) & (df['miss_mass'] > 0.889) & (df['miss_mass'] < 0.989)

# Create a 'Golden' dataframe
df_gold = df[golden_cut]

print(f"Found {len(df_gold)} Antineutrons")
print("Deposited Energies <30deg (Enbar_calo) in GeV:")
print(df_gold['Enbar_calo'].values)
# %%