"""Beam-weighted sky power vs elevation, 2026-07-17 motor raster (key 4).

v2 changes vs beammod_fold.py:
  - channel MEDIAN not mean  -> immune to single-channel RFI
  - narrow bands at named centre frequencies -> shows the frequency trend directly
  - normalised to ZENITH (el=0), not the sweep median -> physical reference
  - one panel, mean per frequency + shaded sweep-to-sweep scatter
"""
import h5py, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

P="/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
OUT="/home/christian/Documents/research/papers/eigsep_instrument/notebooks-s11"
with h5py.File(P,"r") as f:
    d=f["data4"][:].astype(np.float64); t=f["times"][:]
    cpd=f.attrs["counts_per_deg"]
    az=f["az_counts"][:]/cpd; el=f["el_counts"][:]/cpd; fr=f["freqs"][:]
tm=(t-t[0])/60; ch=np.arange(1024); d[d<=0]=np.nan
comb=np.isin(ch%16,[15,0,1])

CENTRES=[60.,80.,120.,160.]; HW=5.0
COLS=["#1b6ca8","#00897b","#c77700","#b3282d"]
EDGES=np.arange(-180,181,6); CENT=0.5*(EDGES[:-1]+EDGES[1:])

CLEAN_MIN=25.0   # afternoon RFI dominates after this
turn=np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
segs=[(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>40 and tm[a]<CLEAN_MIN]
print(f"elevation sweeps used: {len(segs)} (first {CLEAN_MIN:.0f} min of the raster)")
print(f"azimuth coverage: {np.nanmin(az):.0f} to {np.nanmax(az):.0f} deg\n")

fig,ax=plt.subplots(figsize=(3.4,2.6))
print(f"{'centre':>8}  {'depth':>7}  {'scatter':>8}  {'SNR':>5}   (median across channels)")
for c0,col in zip(CENTRES,COLS):
    m=(np.abs(fr-c0)<=HW)&(~comb)
    y=10*np.log10(np.nanmedian(d[:,m],axis=1))
    prof=[]
    for a,b in segs:
        yy=y[a:b]; ee=el[a:b]
        ref=np.nanmedian(yy)                      # sweep median: lower variance than
        if not np.isfinite(ref): continue         # a few-sample zenith reference
        yy=yy-ref
        prof.append([np.nanmedian(yy[(ee>=EDGES[i])&(ee<EDGES[i+1])]) for i in range(len(CENT))])
    prof=np.array(prof)
    mu=np.nanmean(prof,axis=0); sd=np.nanstd(prof,axis=0)
    ax.fill_between(CENT,mu-sd,mu+sd,color=col,alpha=0.18,lw=0)
    ax.plot(CENT,mu,c=col,lw=1.2,label=f"{c0:.0f} MHz")
    depth=np.nanmax(mu)-np.nanmin(mu)
    print(f"{c0:6.0f}MHz  {depth:6.2f}dB  {np.nanmedian(sd):7.2f}dB  {depth/np.nanmedian(sd):5.1f}   n_sweeps={len(prof)}")

ax.axvline(0,c="0.7",lw=0.6,ls=":")
ax.set_xlabel("Elevation from zenith [deg]",fontsize=8)
ax.set_ylabel("Power rel. sweep median [dB]",fontsize=8)
ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180])
ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
ax.legend(fontsize=6.5,ncol=2,loc="lower center",framealpha=0.9)
fig.savefig(f"{OUT}/beam_modulation.pdf",bbox_inches="tight",dpi=600)
fig.savefig(f"{OUT}/beam_modulation.png",bbox_inches="tight",dpi=150)
print("\nwrote beam_modulation.pdf/.png")
