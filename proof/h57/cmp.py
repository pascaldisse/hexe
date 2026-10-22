import re,sys
def load(p):
    r={}
    for l in open(p):
        m=re.match(r'(\S+ \S+)\tenv min (\S+).*self min (\S+)',l)
        if m: r[m.group(1)]=(float(m.group(2)),float(m.group(3)))
    return r
a=load(sys.argv[1]); b=load(sys.argv[2])
for k in a:
    x=a[k]; y=b[k]
    flag='  <-- >2mm' if (y[0]-x[0]<-0.002 or y[1]-x[1]<-0.002) else ''
    if x!=(0,0) or y!=(0,0): print(f"{k:22s} env {x[0]*1000:4.0f}->{y[0]*1000:4.0f}  self {x[1]*1000:4.0f}->{y[1]*1000:4.0f}{flag}")
