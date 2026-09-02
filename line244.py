import glob, json, warnings
import numpy as np, h5py
from scipy.ndimage import median_filter
warnings.filterwarnings("ignore")
NCH=1024; freq=np.arange(NCH)*0.244140625; ch=np.arange(NCH)
CH244=999

def states_2025(h,key):
    sw=json.loads(h["metadata"]["rfswitch"][()])
    st=np.array([d.get("sw_state",-1) for d in sw]); n=h["data"][key].shape[0]
    st=np.pad(st,(0,max(0,n-len(st))),constant_values=-1)[:n]
    return st==0, np.isin(st,[128,160,224])

def states_2026(h,key):
    sw=np.array([str(x) for x in json.loads(h["metadata"]["rfswitch"][()])])
    n=h["data"][key].shape[0]
    sw=np.pad(sw,(0,max(0,n-len(sw))),constant_values="?")[:n]
    return sw=="RFANT", np.isin(sw,["RFNOFF","RFNON","RFAMB","RFSP1_SHORT","RFSP1_OPEN"])

def stack(files,key,statefn,which,maxf=60):
    acc=np.zeros(NCH); n=0
    for fn in files[:maxf]:
        try:
            with h5py.File(fn,"r") as h:
                if key not in h["data"]: continue
                a,l=statefn(h,key); m=a if which=="ant" else l
                if m.sum()==0: continue
                d=np.abs(h["data"][key][:])
                acc+=np.nansum(d[m],axis=0); n+=m.sum()
        except Exception: continue
    return (acc/max(n,1)), n

def excess(spec,c=CH244,w=21):
    db=10*np.log10(spec+1e-30)
    base=median_filter(db,size=w,mode="nearest")
    return db[c]-base[c], db[c], base[c]

EP={
 "2025": (sorted(glob.glob("/media/christian/T7/data/deployment4/corr_data/corr_20250720_0*.h5")),states_2025,{"bowtie":"2","ground":"0"}),
 "2026": (sorted(glob.glob("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered/corr_20260717_0*.h5")),states_2026,{"bowtie":"4","ground":"0"}),
}
print(f"243.90 MHz (ch {CH244}) -- excess over local running median\n")
for ep,(fl,sfn,keys) in EP.items():
    print(f"=== {ep} ===")
    for role,k in keys.items():
        for which in ("ant","load"):
            s,n=stack(fl,k,sfn,which)
            if n==0: print(f"  {role:7s} key {k} {which:4s}: no integrations"); continue
            ex,lv,bs=excess(s)
            print(f"  {role:7s} key {k} {which:4s}: excess {ex:+6.2f} dB | line {lv:6.2f} | baseline {bs:6.2f} | n={n}")
    # continuum level near 244 for floor comparison
    s,_=stack(fl,keys["bowtie"],sfn,"ant")
    nb=(freq>=235)&(freq<=250)&(np.abs(ch-CH244)>3)
    print(f"  continuum 235-250 MHz (bowtie, ant) median raw = {np.median(s[nb]):.4g}")
    lo=(freq>=60)&(freq<=80)
    print(f"  continuum  60-80  MHz (bowtie, ant) median raw = {np.median(s[lo]):.4g}")
    print(f"  band tilt  (60-80)/(235-250)                   = {np.median(s[lo])/np.median(s[nb]):.4g}")
    print()
