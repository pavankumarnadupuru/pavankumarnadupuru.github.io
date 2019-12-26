import struct
import numpy as np
from matplotlib import pyplot as ply

import tensorflow as tf
from tensorflow import keras

filename={}
filename['train_images']='/home/pavan/Desktop/_telugu/train-images-idx3-ubyte'
filename['train_labels']='/home/pavan/Desktop/_telugu/train-labels-idx1-ubyte'
filename['test_images']='/home/pavan/Desktop/_telugu/t10k-images-idx3-ubyte'
filename['test_labels']='/home/pavan/Desktop/_telugu/t10k-labels-idx1-ubyte'

with open(filename['train_images'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    train_images=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

with open(filename['train_labels'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    train_labels=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

with open(filename['test_images'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    test_images=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

with open(filename['test_labels'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    test_labels=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

print(f'The tensorflow version is : ',tf.__version__)
