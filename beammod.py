"""2026-07-17 motor raster: beam-weighted sky power vs platform orientation.

Uses NON-comb channels only, so this is the sky seen through a moving beam --
not a transmitter beam map (that is Vazquez et al.).
"""
import h5py, warnings, numpy as np
warnings.filterwarnings("ignore")
P="/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
with h5py.File(P,"r") as f:
    d=f["data4"][:].astype(np.float64); t=f["times"][:]
    cpd=f.attrs["counts_per_deg"]
    az=f["az_counts"][:]/cpd; el=f["el_counts"][:]/cpd
    elt=f["el_target_counts"][:]/cpd; azt=f["az_target_counts"][:]/cpd
    fr=f["freqs"][:]  # already MHz
tm=(t-t[0])/60; ch=np.arange(1024)
d[d<=0]=np.nan
print("NaN spectra rows:", int(np.isnan(d).all(axis=1).sum()), " NaN cells:", int(np.isnan(d).sum()))

comb = np.isin(ch%16,[15,0,1])          # comb tones + spill, excluded
BANDS={"55-75 MHz":(55,75),"75-95 MHz":(75,95),"110-130 MHz":(110,130),"150-170 MHz":(150,170)}
print("\nband power modulation over the raster (non-comb channels):")
series={}
for lbl,(lo,hi) in BANDS.items():
    m=(fr>=lo)&(fr<=hi)&(~comb)
    y=np.nanmean(d[:,m],axis=1)
    ydb=10*np.log10(y); ydb-=np.nanmedian(ydb)
    series[lbl]=ydb
    print(f"  {lbl:12s} n_ch={m.sum():4d}  pk-pk {np.nanpercentile(ydb,99)-np.nanpercentile(ydb,1):5.2f} dB"
          f"   r(|el|)={np.corrcoef(np.abs(el[np.isfinite(ydb)]),ydb[np.isfinite(ydb)])[0,1]:+.2f}")

# repeatability: bin power vs el for each az pass, then compare passes
lbl="110-130 MHz"; y=series[lbl]
azp=np.round(az/5)*5
passes=[a for a in np.unique(azp) if np.isfinite(a) and (azp==a).sum()>100]
edges=np.arange(-180,181,10); cent=0.5*(edges[:-1]+edges[1:])
prof=[]
for a in passes:
    s=azp==a
    p=[np.nanmedian(y[s&(el>=edges[i])&(el<edges[i+1])]) for i in range(len(cent))]
    prof.append(p)
prof=np.array(prof)
print(f"\nrepeatability across {len(passes)} az passes ({lbl}), power vs elevation:")
print(f"  median scatter between passes at fixed el : {np.nanmedian(np.nanstd(prof,axis=0)):.2f} dB")
print(f"  peak-to-trough of the mean el profile     : {np.nanmax(np.nanmean(prof,axis=0))-np.nanmin(np.nanmean(prof,axis=0)):.2f} dB")
np.savez("/tmp/claude-1000/-home-christian-Documents-research-papers-eigsep-instrument-eigsep-instrument-rasti/e62937a6-8e0e-48ed-a420-2d6219fcae22/scratchpad/beammod.npz",
         tm=tm,az=az,el=el,cent=cent,prof=prof,passes=np.array(passes),
         **{f"s|{k}":v for k,v in series.items()})
