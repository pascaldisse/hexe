import sys,struct,re
d=open(sys.argv[1],'rb').read(); base=0x401000; t=d[0x1000:0x1000+0x502000]
pat=re.compile(rb'(\xd9|\x8b|\xd8|\xdc)[\x80-\xbf]\xb4\x01\x00\x00')
for m in pat.finditer(t): print(hex(base+m.start()), t[m.start():m.start()+6].hex())
