import h5py
import numpy as np
lrr = r'rawdata\wgEncodeHaibTfbsGm12878BatfPcr1xPkRep1\rice\data2hangshuju\datalabel.hdf5'
with h5py.File(lrr, 'r') as f:
    seqs_vector = np.asarray(f['data'])[:52208]
    labels = np.asarray(f['label'])[:52208]
print(type(labels))