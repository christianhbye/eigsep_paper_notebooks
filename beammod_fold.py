import h5py, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
SP="/tmp/claude-1000/-home-christian-Documents-research-papers-eigsep-instrument-eigsep-instrument-rasti/e62937a6-8e0e-48ed-a420-2d6219fcae22/scratchpad"
P="/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
with h5py.File(P,"r") as f:
    d=f["data4"][:].astype(np.float64); t=f["times"][:]
    cpd=f.attrs["counts_per_deg"]
    az=f["az_counts"][:]/cpd; el=f["el_counts"][:]/cpd; fr=f["freqs"][:]
tm=(t-t[0])/60; ch=np.arange(1024); d[d<=0]=np.nan
comb=np.isin(ch%16,[15,0,1])
CLEAN = tm < 25.0                      # before the afternoon RFI

# split into individual elevation sweeps at the turning points
turn=np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
segs=[(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>40]
segs=[s for s in segs if tm[s[0]]<25.0]
print(f"elevation sweeps in the clean window: {len(segs)}")

BANDS={"55-75 MHz":(55,75),"110-130 MHz":(110,130),"150-170 MHz":(150,170)}
edges=np.arange(-180,181,6); cent=0.5*(edges[:-1]+edges[1:])

fig,axes=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
for ax,(lbl,(lo,hi)) in zip(axes,BANDS.items()):
    m=(fr>=lo)&(fr<=hi)&(~comb)
    y=10*np.log10(np.nanmean(d[:,m],axis=1))
    prof=[]
    for a,b in segs:
        yy=y[a:b]-np.nanmedian(y[a:b]); ee=el[a:b]
        p=np.array([np.nanmedian(yy[(ee>=edges[i])&(ee<edges[i+1])]) for i in range(len(cent))])
        prof.append(p); ax.plot(cent,p,lw=0.6,alpha=0.35,c="0.5")
    prof=np.array(prof)
    mean=np.nanmean(prof,axis=0); sc=np.nanstd(prof,axis=0)
    ax.plot(cent,mean,lw=1.6,c="C3")
    ax.set_title(lbl,fontsize=9); ax.set_xlabel("elevation [deg]",fontsize=8)
    ax.grid(alpha=.3); ax.tick_params(labelsize=7); ax.set_xlim(-180,180)
    depth=np.nanmax(mean)-np.nanmin(mean)
    print(f"  {lbl:12s} modulation depth {depth:5.2f} dB   median sweep-to-sweep scatter {np.nanmedian(sc):.2f} dB"
          f"   SNR {depth/np.nanmedian(sc):.1f}")
axes[0].set_ylabel("power [dB rel. sweep median]",fontsize=8)
fig.suptitle("2026-07-17 raster: beam-weighted sky power vs elevation, individual sweeps (grey) and mean (red)",fontsize=9)
fig.tight_layout(); fig.savefig(f"{SP}/beammod_fold.png",dpi=130)
print("saved")
