"""2025 vs 2026 narrowband self-RFI line census on the suspended bowtie.

Like-for-like: antenna-state integrations only, night windows, identical
detrending and identical masking (every 8th channel masked in BOTH epochs so
the 2025 every-16th comb and the 2026 every-8th comb are removed the same way).
"""
import glob, json, warnings
import numpy as np, h5py
from scipy.ndimage import median_filter
warnings.filterwarnings("ignore")

NCH = 1024
freq = np.arange(NCH) * 0.244140625
ch = np.arange(NCH)

def _fit(mask, n):
    """rfswitch reports one entry per integration; pad/trim defensively."""
    if len(mask) < n:
        mask = np.pad(mask, (0, n - len(mask)), constant_values=False)
    return mask[:n]

def ant_rows_2025(h, key):
    # sw_state 0 = antenna; loads are {128, 160, 224}
    sw = json.loads(h["metadata"]["rfswitch"][()])
    st = np.array([d.get("sw_state", -1) for d in sw])
    return _fit(st == 0, h["data"][key].shape[0])

def ant_rows_2026(h, key):
    sw = np.array([str(x) for x in json.loads(h["metadata"]["rfswitch"][()])])
    return _fit(sw == "RFANT", h["data"][key].shape[0])

def night_spectrum(files, key, rowsel, maxfiles=40):
    acc, nrow = np.zeros(NCH), 0
    for fn in files[:maxfiles]:
        try:
            with h5py.File(fn, "r") as h:
                if key not in h["data"]: continue
                d = np.abs(h["data"][key][:])
                m = rowsel(h, key)
                if m.sum() == 0: continue
                acc += np.nansum(d[m], axis=0); nrow += m.sum()
        except Exception:
            continue
    return acc / max(nrow, 1), nrow

def census(spec, label):
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(spec)
    base = median_filter(db, size=21, mode="nearest")
    resid = db - base
    mask = np.ones(NCH, bool)
    mask &= (freq >= 50) & (freq <= 246)          # analysis band
    mask &= (ch % 8) != 0                          # comb: covers 16th AND 8th
    mask &= ~((freq >= 88) & (freq <= 108))        # FM
    lines = mask & (resid > 0.5)
    print(f"  {label}")
    print(f"    channels examined      : {mask.sum()}")
    print(f"    lines > 0.5 dB         : {lines.sum()}")
    print(f"    lines > 1.0 dB         : {(mask & (resid > 1.0)).sum()}")
    print(f"    lines > 2.0 dB         : {(mask & (resid > 2.0)).sum()}")
    print(f"    median |residual|      : {np.median(np.abs(resid[mask])):.3f} dB")
    print(f"    residual 99th pctile   : {np.percentile(resid[mask], 99):.2f} dB")
    top = np.argsort(resid * lines)[::-1][:6]
    print(f"    strongest lines        : " + ", ".join(
        f"{freq[i]:.2f}MHz(ch{i},{resid[i]:+.1f}dB)" for i in top if lines[i]))
    return resid, mask

print("=== 2025 (deployment 4, key 2 -- post-swap bowtie), Jul 20 night ===")
f25 = sorted(glob.glob("/media/christian/T7/data/deployment4/corr_data/corr_20250720_0*.h5"))
s25, n25 = night_spectrum(f25, "2", ant_rows_2025)
print(f"  antenna-state integrations: {n25} from {len(f25)} files")
r25, m25 = census(s25, "2025")

print()
print("=== 2026 (deployment 5, key 4 = box-air), Jul 17 night ===")
f26 = sorted(glob.glob("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered/corr_20260717_0*.h5"))
s26, n26 = night_spectrum(f26, "4", ant_rows_2026)
print(f"  antenna-state integrations: {n26} from {len(f26)} files")
r26, m26 = census(s26, "2026")

np.savez("/tmp/claude-1000/-home-christian-Documents-research-papers-eigsep-instrument-eigsep-instrument-rasti/e62937a6-8e0e-48ed-a420-2d6219fcae22/scratchpad/rfi_compare.npz",
         freq=freq, s25=s25, s26=s26, r25=r25, r26=r26, m25=m25, m26=m26, n25=n25, n26=n26)
