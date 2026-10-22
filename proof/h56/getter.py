import sys,re
d=open('endorphin.exe','rb').read(); base=0x401000; t=d[0x1000:0x1000+0x502000]
for off in sys.argv[1:]:
    o=int(off,16).to_bytes(4,'little')
    for m in re.finditer(re.escape(b'\x8d\x81'+o+b'\xc3'),t): print(off,'getter lea ecx+off; ret',hex(base+m.start()))
    for m in re.finditer(re.escape(o),t):
        a=m.start(); 
        if t[a-1]&0xc0==0x80: print(off,'disp',hex(base+a-2),t[a-2:a+4].hex())
