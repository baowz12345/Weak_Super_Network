#!/usr/bin/bash
#!/usr/bin/bash

Threadnum=3
tmp_file="/tmp/$$.fifo"
mkfifo $tmp_file
exec 6<> $tmp_file
rm $tmp_file
for((i=0; i<$Threadnum; ++i))
do
    echo ""
done >&6

# In the training and testing phase
for eachTF in `ls ./rawdata/Hepg2/data`
do
    read -u6
    {
	echo $eachTF
	python encoding.py ./rawdata/Hepg2/data/positive1.txt ./rawdata/Hepg2/data/negative1.txt ./rawdata/Hepg2/3/datalabel.hdf5 -m ./mappers/3mer.txt -c 50 -s 10 --no-reverse -kmer 3 -run 'ws'

	echo "" >&6
    }&
done
wait
exec 6>&-
