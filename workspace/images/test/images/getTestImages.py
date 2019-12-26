#Done generation of all test images 

import struct
from matplotlib import pyplot as plt
import numpy as np
filename='t10k-images-idx3-ubyte'
with open(filename, 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    arr=np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
name=1570
for i in range(len(arr)):
    if(i<name):
        print('Hahaa')
        continue
    plt.imshow(arr[i],interpolation='nearest')
    temp=str(name)+'.png'
    name+=1
    plt.savefig(temp)


