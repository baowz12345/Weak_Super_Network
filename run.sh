#!/usr/bin/bash


# In the training and testing phase
for eachTF in `ls ./rawdata/`
do 
	echo $eachTF
	if [ -d ./model/$eachTF ]; then
	   echo $eachTF 'has existed.'
	   continue
	fi
	python my_train_val_test1.py -datalable ./rawdata/Gm12878/data_nows1/datalabel.hdf5 -k 3 -batchsize 300 -params 12

done

