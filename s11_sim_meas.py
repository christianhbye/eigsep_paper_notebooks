"""Measured vs simulated antenna reflection coefficient for the EIGSEP bowtie.

Measurement: field OSL calibration, lab-measured S-parameters de-embedded
(MIST approach).  Simulation: HFSS, sparams.csv.
Style follows s11-plot.ipynb so the figure matches the rest of the paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/christian/Documents/research/papers/eigsep_instrument/notebooks"
OUT = "/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti"

d = np.load(f"{HERE}/s11.npz")
f_meas = d["freqs"] / 1e6
s_meas = 20 * np.log10(np.abs(d["calibrated_s11"]))
cut = (f_meas >= 50) & (f_meas <= 250)
f_meas, s_meas = f_meas[cut], s_meas[cut]

sim = np.loadtxt(f"{HERE}/sparams.csv", delimiter=",", skiprows=1, usecols=(1, 2))
f_sim, s_sim = sim[:, 0], sim[:, 1]

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(f_sim, s_sim, c="0.55", ls="--", lw=1.1, label="Simulation")
ax.plot(f_meas, s_meas, c="k", lw=1.0, label="Measurement")
ax.axhline(-10, c="0.75", ls=":", lw=0.8, zorder=0)
ax.set_xlabel("Frequency [MHz]", fontsize=8)
ax.set_ylabel(r"|$\Gamma_{\rm ant}$| [dB]", fontsize=8)
ax.set_ylim(-23, 0)
ax.set_xlim(50, 250)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.4)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
fig.savefig(f"{OUT}/s11_sim_meas.pdf", bbox_inches="tight", dpi=600)

# --- numbers for the manuscript text ---
def stays_below(fr, sd, thr=-10):
    ok = sd < thr
    idx = [i for i in range(len(ok)) if ok[i:].all()]
    return fr[idx[0]] if idx else None

s_sim_i = np.interp(f_meas, f_sim, s_sim)
diff = np.abs(s_meas - s_sim_i)
print(f"measurement stays < -10 dB above : {stays_below(f_meas, s_meas):.1f} MHz")
print(f"simulation  stays < -10 dB above : {stays_below(f_sim, s_sim):.1f} MHz")
print(f"median |meas - sim| over 50-250  : {np.median(diff):.1f} dB")
print(f"max    |meas - sim| over 50-250  : {np.max(diff):.1f} dB  (at {f_meas[np.argmax(diff)]:.0f} MHz)")
for band, lo, hi in [("50-85", 50, 85), ("85-250", 85, 250)]:
    m = (f_meas >= lo) & (f_meas <= hi)
    print(f"  median |meas - sim| {band} MHz    : {np.median(diff[m]):.1f} dB")
print(f"measured |Gamma| range 85-250    : {s_meas[f_meas>=85].min():.1f} to {s_meas[f_meas>=85].max():.1f} dB")
