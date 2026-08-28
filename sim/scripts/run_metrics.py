import sys, numpy as np
d=np.load(sys.argv[1]); ch=d["chainage"]/1000; ps=d["peak_stage"]; t=d["t"]; g=d["Galchhi"]
gorge=np.median(ps[(ch>=33)&(ch<=52)])
m30=0.0
for j in range(len(g)):
    j2=np.searchsorted(t,t[j]+1800)
    if j2<len(g): m30=max(m30,g[j2]-g[j])
print(f"{sys.argv[2]}: gorge_median={gorge:.1f} m, galchhi_30min={m30:.1f} m")
