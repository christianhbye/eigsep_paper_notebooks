import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP="/tmp/claude-1000/-home-christian-Documents-research-papers-eigsep-instrument-eigsep-instrument-rasti/e62937a6-8e0e-48ed-a420-2d6219fcae22/scratchpad"
z=np.load(f"{SP}/raster.npz")
tm,pitch,moving=z["tmin"],z["PITCH"],z["moving"]
def db(a):
    with np.errstate(divide="ignore"): return 10*np.log10(a)
def norm(a): 
    d=db(a); return d-np.nanmedian(d[~moving])

fig,ax=plt.subplots(3,1,figsize=(7,7),sharex=True)
m0,m1=tm[moving].min(),tm[moving].max()
for a in ax: a.axvspan(m0,m1,color="0.9",zorder=0)

ax[0].plot(tm,pitch,c="k",lw=0.8)
ax[0].set_ylabel("Platform pitch [deg]\n(panda IMU)",fontsize=9)
ax[0].set_title("July 19 2025 pitch raster — shaded = platform moving",fontsize=10)

ax[1].plot(tm,norm(z["bowtie (key 2)|125-150"]),c="C3",lw=0.7)
ax[1].set_ylabel("Bowtie 125–150 MHz\n[dB rel. parked]",fontsize=9)
ax[1].axhline(0,c="0.6",lw=0.6,ls=":")

for k,c,l in [("ground viv1-N (key 0)|125-150","C0","key 0"),
              ("ground viv1-E (key 1)|125-150","C2","key 1"),
              ("ground (key 3)|125-150","C1","key 3")]:
    ax[2].plot(tm,norm(z[k]),c=c,lw=0.7,alpha=0.85,label=l)
ax[2].set_ylabel("Ground antennas\n125–150 MHz [dB]",fontsize=9)
ax[2].axhline(0,c="0.6",lw=0.6,ls=":")
ax[2].legend(fontsize=8,ncol=3); ax[2].set_xlabel("Minutes from 12:58 MDT",fontsize=9)
ax[1].set_ylim(-12,12); ax[2].set_ylim(-12,12)
for a in ax: a.tick_params(labelsize=8); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{SP}/raster_prototype.png",dpi=130)
print("saved")
