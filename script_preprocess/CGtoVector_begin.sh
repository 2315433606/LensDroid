#!/bin/bash
for idx in `seq 0 1000 10000`
do
start=$idx
end=$(($idx+1000))
echo $start
	nohup python -u apkPreprocess/script_preprocess/CG2_CGtoVector_multithread.py --start=$start --end=$end > apkPreprocess/process_Log/Vector_log825/CG2drebinBenign_$start.log 2>&1 &
done
