import sys,struct
d=open('endorphin.exe','rb').read()
# .data paddr 0x601000 vaddr 0xa01000 size 0x59000; .rdata paddr 0x504000 vaddr 0x904000
for (po,va,sz) in [(0x601000,0xa01000,0x59000),(0x504000,0x904000,0xfd000)]:
    for t in sys.argv[1:]:
        v=int(t,16); b=struct.pack('<I',v); i=d.find(b,po)
        while i!=-1 and i<po+sz:
            print(t,'@',hex(va+i-po)); i=d.find(b,i+1)
