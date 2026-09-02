"""July 19 2025 pitch raster: does received power track platform orientation,
and does it do so in the STATIONARY ground antennas as well as the bowtie?

Ground antennas cannot have moved, so synchronous modulation there is the
discriminator that does not require knowing whether the bowtie moved.
"""
import glob, json, warnings
import numpy as np, h5py
warnings.filterwarnings("ignore")
NCH=1024; freq=np.arange(NCH)*0.244140625

FILES = sorted(glob.glob("/media/christian/T7/data/deployment4/corr_data/corr_20250719_12*.h5"))
KEYS = {"bowtie (key 2)":"2","ground viv1-N (key 0)":"0","ground viv1-E (key 1)":"1","ground (key 3)":"3"}
BANDS = {"60-80":(60,80),"95-110":(95,110),"125-150":(125,150)}

T=[]; PITCH=[]; SER={k:{b:[] for b in BANDS} for k in KEYS}
for fn in FILES:
    try:
        with h5py.File(fn,"r") as h:
            t=np.array(json.loads(h["header"]["times"][()]))
            pa=json.loads(h["metadata"]["imu_panda"][()])
            ax=np.array([d.get("accel_x",np.nan) for d in pa])
            ay=np.array([d.get("accel_y",np.nan) for d in pa])
            az=np.array([d.get("accel_z",np.nan) for d in pa])
            p=np.degrees(np.arctan2(ax,np.hypot(ay,az)))
            sw=json.loads(h["metadata"]["rfswitch"][()])
            st=np.array([d.get("sw_state",-1) for d in sw])
            n=min(len(t),len(p),len(st),h["data"]["2"].shape[0])
            ant=(st[:n]==0)
            if ant.sum()==0: continue
            T.append(t[:n][ant]); PITCH.append(p[:n][ant])
            for lbl,k in KEYS.items():
                d=np.abs(h["data"][k][:n][ant])
                for b,(lo,hi) in BANDS.items():
                    m=(freq>=lo)&(freq<=hi)
                    SER[lbl][b].append(np.nanmean(d[:,m],axis=1))
    except Exception:
        continue

T=np.concatenate(T); PITCH=np.concatenate(PITCH)
o=np.argsort(T); T=T[o]; PITCH=PITCH[o]
for lbl in KEYS:
    for b in BANDS:
        SER[lbl][b]=np.concatenate(SER[lbl][b])[o]

t0=T[0]; tmin=(T-t0)/60.0
print(f"integrations: {len(T)}   span: {tmin.max():.1f} min")
print(f"pitch range : {np.nanmin(PITCH):.0f} to {np.nanmax(PITCH):.0f} deg")
moving = np.abs(np.gradient(PITCH)) > 0.5
print(f"moving samples: {moving.sum()}  parked: {(~moving).sum()}")
print()
print("Correlation of band power with |pitch|, and modulation depth (moving only):")
print(f"  {'series':24s} {'band':9s} {'r(|pitch|)':>11s} {'pk-pk [dB]':>11s} {'parked pk-pk':>13s}")
for lbl in KEYS:
    for b in BANDS:
        y=SER[lbl][b]
        with np.errstate(divide="ignore"): ydb=10*np.log10(y)
        good=np.isfinite(ydb)&moving
        pk=np.nan; r=np.nan
        if good.sum()>20:
            r=np.corrcoef(np.abs(PITCH[good]),ydb[good])[0,1]
            pk=np.percentile(ydb[good],97.5)-np.percentile(ydb[good],2.5)
        gp=np.isfinite(ydb)&(~moving)
        pkp=np.percentile(ydb[gp],97.5)-np.percentile(ydb[gp],2.5) if gp.sum()>20 else np.nan
        print(f"  {lbl:24s} {b:9s} {r:11.2f} {pk:11.2f} {pkp:13.2f}")

np.savez("/tmp/claude-1000/-home-christian-Documents-research-papers-eigsep-instrument-eigsep-instrument-rasti/e62937a6-8e0e-48ed-a420-2d6219fcae22/scratchpad/raster.npz",
         T=T,tmin=tmin,PITCH=PITCH,moving=moving,
         **{f"{l}|{b}":SER[l][b] for l in KEYS for b in BANDS})
