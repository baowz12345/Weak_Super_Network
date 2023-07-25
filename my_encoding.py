import argparse,os,numpy as np,h5py,sys
from os.path import exists,dirname
from os import makedirs
#from itertools import izip
from numpy.random.tests.test_direct import pwd
from tqdm import tqdm
izip = zip

from sklearn.decomposition import PCA





def parse_args():
    #创建解析器
    parser = argparse.ArgumentParser(description="process is begining.")
    #positional arguments:
    parser.add_argument("posfile", type=str, help="Positive sequences in FASTA/TSV format (with .fa/.fasta or .tsv extension)", default=r'D:\workship\anewlife\wscnnlstmchange\rawdata\wgEncodeHaibTfbsGm12878BatfPcr1xPkRep1\rice\data5hangshuju\negative.fasta')
    parser.add_argument("negfile", type=str, help=r"Negative sequences in FASTA/TSV format", default=r'D:\workship\anewlife\wscnnlstmchange\rawdata\wgEncodeHaibTfbsGm12878BatfPcr1xPkRep1\rice\data5hangshuju\positive.fasta')
    parser.add_argument("outfile", type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.hdf5). ", default=r'D:\workship\anewlife\wscnnlstmchange\rawdata\wgEncodeHaibTfbsGm12878BatfPcr1xPkRep1\rice\data5hangshuju')
    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="", help="A file mapping each nucleotide to a vector.")

    parser.add_argument("-kmer", "--kmer", dest="kmer", type=int, default=1, help="the length of kmer")
    # parser.add_argument("-run", "--run", dest="run", type=str, default='ws', help="order")

    return parser.parse_args()

def Load_mapper(mapperfile):
    mapper = {}
    with open(mapperfile,'r') as f:
        for x in f:
            # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            #通过指定分隔符对字符串进行分割并返回一个列表，默认分隔符为所有空字符，包括空格、换行(\n)、制表符(\t)等.
            line = x.strip().split()
            # lower() 方法转换字符串中所有大写字符为小写。
            #line[0]去掉指定的一个，第一个
            word = line[0].lower()
            vec = [float(item) for item in line[1:]]
            mapper[word] = vec
    return mapper
def convert(pos_file, neg_file, outfile, mapper, kmer):
    with open(pos_file) as posf,open(neg_file) as negf:
        pos_data = posf.readlines(); neg_data = negf.readlines()

    pos_seqs = [];neg_seqs = []
    for line in pos_data:
        if '>' not in line:
           fw_seq = list(line.strip().lower())
           pos_seqs.append(fw_seq)
    for line in neg_data:
        if '>' not in line:
           fw_seq = list(line.strip().lower())
           neg_seqs.append(fw_seq)
    seqs = pos_seqs + neg_seqs
    #生成相应的标签
    label = np.asarray([1] * len(pos_seqs) + [0] * len(neg_seqs))
    print(label)
    #if run == 'ws':
    seqs_vector = []

    if kmer == 1:
        for seq in seqs:
            mat = [mapper[element] for element in seq if element in mapper]
            seqs_vector.append(mat)
            #return  seqs_vector
    else:
        print('mistack')

    seqs_vector = np.asarray(seqs_vector)
    outputHDF5(seqs_vector, label, outfile)

def outputHDF5(data,label,filename,labelname='label',dataname='data'):
    print(('data shape: ',data.shape))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    label = [[x.astype(np.float32)] for x in label]
    with h5py.File(filename, 'w',libver='latest') as f:
        f.create_dataset(dataname, data=data, **comp_kwargs)
        f.create_dataset(labelname, data=label, **comp_kwargs)


if __name__ == "__main__":
    args = parse_args()
    outdir = dirname(args.outfile)

    if not exists(outdir):
        makedirs(outdir)

    if args.mapperfile == "":
        print('using one-hot encoding.')
        args.mapper = {'a':[1,0,0,0], 'c':[0,1,0,0], 'g':[0,0,1,0], 't':[0,0,0,1], 'n':[0,0,0,0]}
    else:
        #其实就是format()后面的内容，填入大括号中（可以按位置，或者按变量）
        print('using {} encoding'.format(args.mapperfile))
        args.mapper = Load_mapper(args.mapperfile)
    convert(args.posfile, args.negfile, args.outfile, args.mapper, args.kmer)
