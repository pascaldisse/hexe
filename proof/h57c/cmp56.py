# usage: cmp56.py h56 h57 "h56pert,..." "h57pert,..." -> per row: spread56 (h56+h56pert), spread57 (h57+h57pert),
# tol = max(2mm, spread56, spread57), d = h57-h56; ok d>=-2mm | noise |d|<=tol | REGRESS
import re,sys
def load(p):
    r={}
    for l in open(p):
        m=re.match(r'(\S+ \S+)\tenv min (\S+).*self min (\S+)',l)
        if m: r[m.group(1)]=(float(m.group(2)),float(m.group(3)))
    return r
a=load(sys.argv[1]); b=load(sys.argv[2])
pa=[load(p) for p in sys.argv[3].split(',')]; pb=[load(p) for p in sys.argv[4].split(',')]
print("row                    col   h56  h56pert         sp56   h57  h57pert         sp57  |d|  tol  verdict (mm)")
for k in a:
    for i,c in enumerate(("env","self")):
        va=[a[k][i]]+[p[k][i] for p in pa]; vb=[b[k][i]]+[p[k][i] for p in pb]
        if max(map(abs,va+vb))==0: continue
        s56=max(va)-min(va); s57=max(vb)-min(vb); d=b[k][i]-a[k][i]; tol=max(0.002,s56,s57)
        v='ok' if d>=-0.002 else ('noise' if -d<=tol else 'REGRESS')
        f=lambda xs:' '.join(f'{x*1e3:4.0f}' for x in xs)
        print(f"{k:22s} {c:4s} {va[0]*1e3:5.0f}  {f(va[1:])}  {s56*1e3:5.0f} {vb[0]*1e3:5.0f}  {f(vb[1:])}  {s57*1e3:5.0f} {abs(d)*1e3:4.0f} {tol*1e3:4.0f}  {v}")
