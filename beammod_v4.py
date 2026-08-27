"""Candidate figure: sky vs transmitter-comb modulation over the elevation raster.

2026-07-17 motor raster, key 4 (suspended bowtie). el = 0 is zenith, +/-180 nadir
(pointing at the ground transmitter). 22 elevation sweeps, first 25 min, az -180..+80.
Comb is every 16th channel at OFFSET 8 in 2026 (offset 0 in 2025).
No uncertainty envelope -- diagnostic stage.
"""
import h5py, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
P="/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
OUT="/home/christian/Documents/research/papers/eigsep_instrument/notebooks-s11"
with h5py.File(P,"r") as f:
    d=f["data4"][:].astype(np.float64); t=f["times"][:]
    cpd=f.attrs["counts_per_deg"]; el=f["el_counts"][:]/cpd; az=f["az_counts"][:]/cpd; fr=f["freqs"][:]
tm=(t-t[0])/60; ch=np.arange(1024); d[d<=0]=np.nan
OFF=8
COMB=np.isin(ch%16,[OFF-1,OFF,OFF+1]); TONES=(ch%16)==OFF
E=np.arange(-180,181,6); C=0.5*(E[:-1]+E[1:])
turn=np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
clean=[(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>40 and tm[a]<25]
def prof(y):
    p=[]
    for a,b in clean:
        yy=y[a:b]-np.nanmedian(y[a:b]); ee=el[a:b]
        p.append([np.nanmedian(yy[(ee>=E[i])&(ee<E[i+1])]) for i in range(len(C))])
    return np.nanmean(np.array(p),axis=0)

BANDS=[(55,75),(110,130),(150,170)]; COLS=["#1b6ca8","#c77700","#b3282d"]
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),sharex=True)
for (lo,hi),c in zip(BANDS,COLS):
    lbl=f"{lo}–{hi} MHz"
    ax[0].plot(C,prof(10*np.log10(np.nanmedian(d[:,(fr>=lo)&(fr<=hi)&(~COMB)],axis=1))),c=c,lw=1.3,label=lbl)
    ax[1].plot(C,prof(10*np.log10(np.nanmean (d[:,(fr>=lo)&(fr<=hi)&TONES ],axis=1))),c=c,lw=1.3,label=lbl)
for i,(a,ttl) in enumerate(zip(ax,["(a) Diffuse sky, comb channels masked",
                                   "(b) Transmitter comb channels"])):
    for x in (-90,90): a.axvline(x,c="0.85",lw=0.7,ls="--",zorder=0)
    a.axvline(0,c="0.85",lw=0.7,ls=":",zorder=0)
    a.set_title(ttl,fontsize=8.5)
    a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180])
    a.set_xlabel("Elevation from zenith [deg]",fontsize=8)
    a.tick_params(labelsize=7); a.grid(alpha=0.25)
    a.legend(fontsize=6.5,loc="upper center",framealpha=0.9,ncol=1)
ax[0].set_ylabel("Power rel. sweep median [dB]",fontsize=8)
ax[1].annotate("horizon",xy=(-90,ax[1].get_ylim()[0]),xytext=(-88,ax[1].get_ylim()[0]+0.4),
               fontsize=6,color="0.45")
ax[1].annotate("transmitter\ndirection",xy=(180,0),xytext=(120,-3.2),fontsize=6,color="0.45")
fig.tight_layout()
fig.savefig(f"{OUT}/beam_modulation.pdf",bbox_inches="tight",dpi=600)
fig.savefig(f"{OUT}/beam_modulation.png",bbox_inches="tight",dpi=150)
print(f"sweeps={len(clean)}  az {np.nanmin(az):.0f}..{np.nanmax(az):.0f} deg")
for (lo,hi) in BANDS:
    s=prof(10*np.log10(np.nanmedian(d[:,(fr>=lo)&(fr<=hi)&(~COMB)],axis=1)))
    c_=prof(10*np.log10(np.nanmean(d[:,(fr>=lo)&(fr<=hi)&TONES],axis=1)))
    print(f"  {lo}-{hi} MHz: sky {np.nanmax(s)-np.nanmin(s):.2f} dB   comb {np.nanmax(c_)-np.nanmin(c_):.2f} dB")
