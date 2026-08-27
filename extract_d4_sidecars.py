"""Extract small standalone sidecars from the deployment-4 (July 2025) data on
the T7 external drive, so every 2025 number in the paper can be reproduced
without the drive mounted.

Run once with T7 connected.  Everything downstream reads the .npz files.
"""
import glob, json, warnings
from pathlib import Path
import numpy as np, h5py
warnings.filterwarnings("ignore")

T7 = "/media/christian/T7/data/deployment4"
OUT = "/home/christian/Documents/research/papers/eigsep_instrument/notebooks-s11"
NCH = 1024
freq = np.arange(NCH) * 0.244140625

def switch(h, key):
    sw = json.loads(h["metadata"]["rfswitch"][()])
    st = np.array([d.get("sw_state", -1) for d in sw])
    n = h["data"][key].shape[0]
    st = np.pad(st, (0, max(0, n - len(st))), constant_values=-1)[:n]
    return st == 0, np.isin(st, [128, 160, 224])   # antenna, load

def stack(files, key, which, maxf=200):
    acc = np.zeros(NCH); n = 0
    for fn in files[:maxf]:
        try:
            with h5py.File(fn, "r") as h:
                if key not in h["data"]: continue
                a, l = switch(h, key)
                m = a if which == "ant" else l
                if m.sum() == 0: continue
                acc += np.nansum(np.abs(h["data"][key][:][m]), axis=0); n += int(m.sum())
        except Exception:
            continue
    return acc / max(n, 1), n

# ---- A + B: night spectra, antenna and load, all autocorr keys -------------
night = sorted(glob.glob(f"{T7}/corr_data/corr_20250720_0*.h5"))
spec = {}; counts = {}
for key in ["0", "1", "2", "3", "4"]:
    for which in ("ant", "load"):
        s, n = stack(night, key, which)
        spec[f"{key}_{which}"] = s; counts[f"{key}_{which}"] = n
        print(f"  key {key} {which:4s}: {n:6d} integrations")

# ---- C: the 13 lift S11 sweeps --------------------------------------------
import sys
sys.path.insert(0, "/home/christian/Documents/research/eigsep/data-analysis/src")
from eigsep_data import S11
sf = [Path(f) for f in sorted(glob.glob(f"{T7}/s11_data/ant*.h5"))]
s11 = np.array([S11(f) for f in sf])
s11 = s11[np.argsort([s.timestamp for s in s11])]
lift = s11[:13]
lift_s11 = np.array([s.s11_cal["ant"] for s in lift])
lift_t = np.array([s.timestamp for s in lift])
lift_f = lift[0].freqs
print(f"  lift sweeps: {lift_s11.shape}, {lift_f.min():.0f}-{lift_f.max():.0f} MHz")

np.savez_compressed(
    f"{OUT}/d4_sidecar.npz",
    freq=freq, counts=json.dumps(counts),
    lift_freq=lift_f, lift_s11=lift_s11, lift_times=lift_t,
    note=("deployment 4 (July 2025). Night spectra 2025-07-20 early hours, "
          "antenna (sw_state 0) and load (128/160/224) states, autocorr keys 0-4. "
          "Bowtie is key 4 before the 2025-07-18 18:00 input swap and key 2 after; "
          "this window is post-swap, so key 2 is the bowtie. Key 5 is dead. "
          "Comb tones sit on every 16th channel, offset 0. "
          "lift_s11 = 13 calibrated antenna sweeps during the 2025-07-19 lift."),
    **{f"spec_{k}": v for k, v in spec.items()})
import os
print(f"\nwrote d4_sidecar.npz  ({os.path.getsize(f'{OUT}/d4_sidecar.npz')/1e6:.2f} MB)")
