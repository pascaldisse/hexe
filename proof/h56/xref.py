import sys,struct
d=open('endorphin.exe','rb').read()
base=0x401000; off=0x1000; n=0x502000
t=d[off:off+n]
tg=[int(a,16) for a in sys.argv[1:]]
for i in range(len(t)-5):
    if t[i]==0xE8 or t[i]==0xE9:
        r=struct.unpack_from('<i',t,i+1)[0]; dst=base+i+5+r
        if dst in tg: print(hex(base+i),'call' if t[i]==0xE8 else 'jmp',hex(dst))
    v=struct.unpack_from('<I',t,i)[0]
    if v in tg and i>0: print(hex(base+i-1),'imm',hex(v))
