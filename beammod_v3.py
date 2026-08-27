"""Sky vs transmitter-comb modulation over the 2026-07-17 elevation raster.

NOTE the comb phase differs between epochs:
  2025 (deployment 4): every 16th channel, offset 0
  2026 (deployment 5): every 16th channel, offset 8      <-- masking offset 0 here
                                                             leaves every tone in
"""
import h5py, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
P="/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
OUT="/home/christian/Documents/research/papers/eigsep_instrument/notebooks-s11"
with h5py.File(P,"r") as f:
    d=f["data4"][:].astype(np.float64); t=f["times"][:]
    cpd=f.attrs["counts_per_deg"]; el=f["el_counts"][:]/cpd; fr=f["freqs"][:]
tm=(t-t[0])/60; ch=np.arange(1024); d[d<=0]=np.nan
COMB_OFF=8
COMB_MASK=np.isin(ch%16,[COMB_OFF-1,COMB_OFF,COMB_OFF+1])
TONES=(ch%16)==COMB_OFF
E=np.arange(-180,181,6); C=0.5*(E[:-1]+E[1:])
turn=np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
clean=[(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>40 and tm[a]<25]
def prof(y):
    p=[]
    for a,b in clean:
        yy=y[a:b]-np.nanmedian(y[a:b]); ee=el[a:b]
        p.append([np.nanmedian(yy[(ee>=E[i])&(ee<E[i+1])]) for i in range(len(C))])
    return np.nanmean(np.array(p),axis=0)

BANDS=[(55,75),(110,130),(150,170)]
COLS=["#1b6ca8","#c77700","#b3282d"]
fig,ax=plt.subplots(1,2,figsize=(8,3),sharex=True)
for (lo,hi),c in zip(BANDS,COLS):
    sky=(fr>=lo)&(fr<=hi)&(~COMB_MASK)
    cmb=(fr>=lo)&(fr<=hi)&TONES
    ax[0].plot(C,prof(10*np.log10(np.nanmedian(d[:,sky],axis=1))),c=c,lw=1.3,label=f"{lo}–{hi} MHz")
    ax[1].plot(C,prof(10*np.log10(np.nanmean(d[:,cmb],axis=1))),c=c,lw=1.3,label=f"{lo}–{hi} MHz")
ax[0].set_title("Diffuse sky (comb masked)",fontsize=9)
ax[1].set_title("Transmitter comb tones",fontsize=9)
for a in ax:
    a.axvline(0,c="0.7",lw=0.6,ls=":"); a.grid(alpha=0.3)
    a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180])
    a.set_xlabel("Elevation from zenith [deg]",fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("Power rel. sweep median [dB]",fontsize=8)
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/beammod_sky_vs_comb.png",dpi=140)
print("saved beammod_sky_vs_comb.png")
