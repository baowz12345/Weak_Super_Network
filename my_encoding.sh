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
for eachTF in `ls ./rawdata/`
do
    read -u6
    {
	echo $eachTF
	python encoding.py ./rawdata/H1hesc/1bi1/positive.txt ./rawdata/H1hesc/1bi1/negative.txt ./rawdata/H1hesc/1bi1/datalabel.hdf5 -m ./mappers/1mer.txt -kmer 1
	
	echo "" >&6
    }&
done
wait
exec 6>&-
