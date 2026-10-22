# scan .text for C7 mov [reg+disp8/32], imm32 where imm looks like a float; args: lo hi (va)
import sys,struct
d=open(sys.argv[1],'rb').read(); base=0x401000; off=0x1000; n=0x502000; t=d[off:off+n]
lo,hi=int(sys.argv[2],16),int(sys.argv[3],16)
want=set(int(x,16) for x in sys.argv[4:])
for i in range(lo-base,hi-base):
    if t[i]!=0xC7: continue
    m=t[i+1]; mod=m>>6; rm=m&7
    if (m>>3)&7: continue
    j=i+2
    if rm==4: j+=1
    if mod==1: disp=struct.unpack_from('<b',t,j)[0]; j+=1
    elif mod==2: disp=struct.unpack_from('<i',t,j)[0]; j+=4
    elif mod==0 and rm!=5: disp=0
    else: continue
    v=struct.unpack_from('<I',t,j)[0]
    if v in want: print(hex(base+i),'disp',hex(disp&0xffffffff),hex(v),struct.unpack('<f',struct.pack('<I',v))[0])
