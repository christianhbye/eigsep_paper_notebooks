"""Reproduce every deployment-4 number the paper may cite, using ONLY
d4_sidecar.npz.  No external drive required.
"""
import json, warnings
import numpy as np
from scipy.ndimage import median_filter
warnings.filterwarnings("ignore")

Z = np.load("/home/christian/Documents/research/papers/eigsep_instrument/notebooks-s11/d4_sidecar.npz",
            allow_pickle=True)
freq = Z["freq"]; ch = np.arange(len(freq))
S = {k[5:]: Z[k] for k in Z.files if k.startswith("spec_")}
print(str(Z["note"])[:90], "...\n")

# --- 1. transmitter comb, tone-to-continuum vs frequency --------------------
band = (freq >= 50) & (freq <= 250)
tone = band & (ch % 16 == 0)
near = band & (~np.isin(ch % 16, [15, 0, 1]))
s = S["2_ant"]
print("1. Transmitter comb (key 2, bowtie), tone/continuum:")
print(f"   band median 50-250 MHz : {10*np.log10(np.median(s[tone])/np.median(s[near])):5.1f} dB")
for lo, hi in [(50,100),(100,150),(150,200),(200,250)]:
    t = tone & (freq>=lo) & (freq<hi); n = near & (freq>=lo) & (freq<hi)
    print(f"   {lo:3d}-{hi:3d} MHz          : {10*np.log10(np.median(s[t])/np.median(s[n])):5.1f} dB")

# --- 2. per-key comb strength (which input is the bowtie) -------------------
print("\n2. Comb strength by key (identifies the suspended antenna):")
for k in ["0","1","2","3","4"]:
    v = S[f"{k}_ant"]
    print(f"   key {k}: {10*np.log10(np.median(v[tone])/np.median(v[near])):5.1f} dB")

# --- 3. 243.9 MHz load test ------------------------------------------------
def excess(spec, c=999, w=21):
    db = 10*np.log10(spec + 1e-30)
    return db[c] - median_filter(db, size=w, mode="nearest")[c]
print("\n3. 243.90 MHz (ch 999) excess over local median:")
for k in ["2","0"]:
    print(f"   key {k}: antenna {excess(S[f'{k}_ant']):+6.2f} dB   load {excess(S[f'{k}_load']):+6.2f} dB")

# --- 4. narrowband line census --------------------------------------------
db = 10*np.log10(S["2_ant"]); resid = db - median_filter(db, size=21, mode="nearest")
mask = (freq>=50)&(freq<=246)&((ch%8)!=0)&~((freq>=88)&(freq<=108))
print(f"\n4. Line census (key 2, night, non-comb, non-FM, 50-246 MHz):")
for th in (0.5, 1.0, 2.0):
    print(f"   lines > {th} dB : {(mask & (resid>th)).sum()}")
top = np.argsort(np.where(mask, resid, -99))[::-1][:4]
print("   strongest:", ", ".join(f"{freq[i]:.2f}MHz({resid[i]:+.1f}dB)" for i in top))

# --- 5. lift delay spectrum ------------------------------------------------
lf = Z["lift_freq"]; ls = Z["lift_s11"]
cut = (lf >= 50) & (lf <= 250)
def norm_dly(x):
    xb = x[cut]; w = np.blackman(len(xb))
    dn = np.fft.fftshift(np.fft.fftfreq(len(xb), (lf[cut][1]-lf[cut][0])*1e6))*1e9
    ds = np.fft.fftshift(np.abs(np.fft.fft(xb*w)))
    return dn, ds/ds[(dn>=-20)&(dn<=20)].max()
def band_max(x, lo, hi):
    dn, ds = norm_dly(x); return ds[(dn>=lo)&(dn<hi)].max()
g = np.mean([band_max(ls[i],50,150) for i in (0,1)])
f_ = np.mean([band_max(ls[i],50,150) for i in (10,11,12)])
print(f"\n5. Lift (13 sweeps), reflection/direct in 50-150 ns:")
print(f"   on ground   : {g:.2e}")
print(f"   full height : {f_:.2e}")
print(f"   suppression : {g/f_:.0f}x ({20*np.log10(g/f_):.0f} dB)")
print(f"   floor >300ns at full height: {np.mean([band_max(ls[i],300,1500) for i in (10,11,12)]):.1e}")
