import sys,struct
d=open('endorphin.exe','rb').read()
secs=[(0x601000,0xa01000,0x59000),(0x504000,0x904000,0xfd000)]
def find(v):
    out=[];b=struct.pack('<I',v)
    for po,va,sz in secs:
        i=d.find(b,po)
        while i!=-1 and i<po+sz: out.append(va+i-po); i=d.find(b,i+1)
    return out
def rd(va):
    for po,v0,sz in secs:
        if v0<=va<v0+sz: return struct.unpack_from('<I',d,po+va-v0)[0]
for name in sys.argv[1:]:
    s=d.find(name.encode()+b'\0')
    for po,va,sz in secs:
        if po<=s<po+sz: td=va+s-po-8
    for col in find(td):
        c=col-0xc
        if rd(c)!=0: continue
        for vtm in find(c):
            vt=vtm+4
            print(name,'COL',hex(c),'off',hex(rd(c+4)),'vtable',hex(vt),[hex(rd(vt+4*k)) for k in range(14)])
