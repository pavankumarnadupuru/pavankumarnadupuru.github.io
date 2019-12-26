#Done till image 212
import struct
from matplotlib import pyplot as plt
import numpy as np
filename='train-images-idx3-ubyte'
with open(filename, 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    arr=np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


fi=open('till.txt','r')
fi.seek(0)
name=int(fi.read())
fi.close()
del(fi)


done=0
for i in range(len(arr)):
    if(i<name):
#        print(f' ',i,'is done.')
        continue
    plt.imshow(arr[i],interpolation='nearest')
    temp=str(name)+'.png'
    name+=1
    plt.savefig(temp)
    print(f' ',i,'is done.')
    done+=1
    if(done==25):
        fi=open('till.txt','w+')
        fi.write(str(i))
        fi.close()
        del(fi)
        print('------------------------------------------------')
        exit(0)
#    done+=1
#    if(done==100):
#       print(f'Change name variable to ',name,' and restart process')
#        exit(0)


