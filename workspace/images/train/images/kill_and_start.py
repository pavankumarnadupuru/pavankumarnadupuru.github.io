import subprocess
f=open('till.txt','r')
f.seek(0)
n=int(f.read())
f.close()
del(f)
while(n<4800):
	cmd=['python','getTestImages.py']
	subprocess.Popen(cmd).wait()
