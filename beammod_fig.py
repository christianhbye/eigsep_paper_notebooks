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
m=(fr>=110)&(fr<=130)&(~comb)
y=10*np.log10(np.nanmean(d[:,m],axis=1)); y-=np.nanmedian(y)

fig=plt.figure(figsize=(11,8))
a0=fig.add_subplot(3,1,1)
a0.plot(tm,el,lw=0.5,c="C0",label="el"); a0.plot(tm,az,lw=0.9,c="C3",label="az")
a0.set_ylabel("angle [deg]"); a0.legend(fontsize=8,ncol=2); a0.grid(alpha=.3)
a0.set_title("2026-07-17 motor raster — key 4 (suspended bowtie), 110–130 MHz non-comb")

a1=fig.add_subplot(3,1,2)
a1.plot(tm,y,lw=0.5,c="k"); a1.set_ylabel("power [dB rel. median]")
a1.grid(alpha=.3); a1.set_xlabel("minutes since scan start"); a1.set_ylim(-8,8)

# 2D: power vs (az, el)
a2=fig.add_subplot(3,1,3)
ae=np.arange(-182.5,85,5); ee=np.arange(-180,181,5)
H=np.full((len(ee)-1,len(ae)-1),np.nan)
for i in range(len(ee)-1):
    for j in range(len(ae)-1):
        s=(el>=ee[i])&(el<ee[i+1])&(az>=ae[j])&(az<ae[j+1])
        if s.sum()>2: H[i,j]=np.nanmedian(y[s])
im=a2.pcolormesh(ae[:-1],ee[:-1],H,cmap="plasma",vmin=-3,vmax=3)
a2.set_xlabel("azimuth [deg]"); a2.set_ylabel("elevation [deg]")
fig.colorbar(im,ax=a2,label="dB rel. median")
fig.tight_layout(); fig.savefig(f"{SP}/beammod_prototype.png",dpi=120)
filled=np.isfinite(H).sum()
print(f"grid cells filled: {filled}/{H.size}")
print(f"map dynamic range (5-95 pct): {np.nanpercentile(H,5):.2f} .. {np.nanpercentile(H,95):.2f} dB")
