"""Delay spectrum of the measured antenna reflection coefficient during the
July 2025 lift (deployment 4).  Sweeps at 5 min cadence; the first two were
taken with the antenna still on the ground.

Shows the environmental reflection dropping below the measurement floor as
the antenna is raised -- the measured counterpart of the simulated
environmental delay spectrum in reflections.pdf.
"""
import sys, warnings
sys.path.insert(0, "/home/christian/Documents/research/eigsep/data-analysis/src")
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eigsep_data import S11

OUT = "/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti"
d = Path("/media/christian/T7/data/deployment4/s11_data/")
files = [f for f in d.iterdir() if f.name.startswith("ant")]
s11 = np.array([S11(f) for f in files])
s11 = s11[np.argsort([s.timestamp for s in s11])]
lift = s11[:13]
t0 = lift[0].timestamp

fig, ax = plt.subplots(figsize=(3.4, 2.5))
cmap = plt.get_cmap("plasma")
for i, s in enumerate(lift):
    dn, ds = s.dlys, s.s11_dly["ant"]
    ds = ds / ds[(dn >= -20) & (dn <= 20)].max()
    sel = (dn >= 0) & (dn <= 1000)
    col = cmap(i / (len(lift) - 1) * 0.85)
    lab = None
    if i == 0:
        lab = "on ground"
    elif i == len(lift) - 1:
        lab = "full height"
    ax.plot(dn[sel], ds[sel], color=col, lw=0.9, alpha=0.85, label=lab)

ax.set_yscale("log")
ax.set_xlabel("Delay [ns]", fontsize=8)
ax.set_ylabel("Reflection / direct", fontsize=8)
ax.set_xlim(0, 1000)
ax.set_ylim(1e-5, 1e-1)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.4)
ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
fig.savefig(f"{OUT}/s11_lift.pdf", bbox_inches="tight", dpi=600)

# numbers
def band(s, lo, hi):
    dn, ds = s.dlys, s.s11_dly["ant"]
    ds = ds / ds[(dn >= -20) & (dn <= 20)].max()
    return ds[(dn >= lo) & (dn < hi)].max()

g = np.mean([band(s, 50, 150) for s in lift[:2]])
f_ = np.mean([band(s, 50, 150) for s in lift[10:]])
print(f"50-150 ns, on ground     : {g:.2e}")
print(f"50-150 ns, full height   : {f_:.2e}")
print(f"suppression              : {g/f_:.0f}x  ({20*np.log10(g/f_):.0f} dB)")
print(f">300 ns floor, full height: {np.mean([band(s,300,1500) for s in lift[10:]]):.1e}")
