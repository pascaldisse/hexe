# usage: noise.py h56 h57 pert... -> per row: h56, h57, perturbed values, spread (max-min over h57+pert), verdict
import re,sys
def load(p):
    r={}
    for l in open(p):
        m=re.match(r'(\S+ \S+)\tenv min (\S+).*self min (\S+)',l)
        if m: r[m.group(1)]=(float(m.group(2)),float(m.group(3)))
    return r
a=load(sys.argv[1]); b=load(sys.argv[2]); ps=[load(p) for p in sys.argv[3:]]
print("row                    col   h56   h57  perturbed        spread  |d|  verdict (mm)")
for k in a:
    for i,c in enumerate(("env","self")):
        v=[b[k][i]]+[p[k][i] for p in ps]
        if a[k][i]==0 and max(map(abs,v))==0: continue
        sp=max(v)-min(v); d=b[k][i]-a[k][i]
        tol=max(0.002,sp)
        ok='ok' if d>=-0.002 else ('noise' if -d<=tol else 'REGRESS')
        print(f"{k:22s} {c:4s} {a[k][i]*1e3:5.0f} {b[k][i]*1e3:5.0f}  {' '.join(f'{x*1e3:4.0f}' for x in v[1:])}  {sp*1e3:6.0f} {abs(d)*1e3:4.0f}  {ok}")
