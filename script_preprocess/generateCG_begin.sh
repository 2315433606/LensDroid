#!/bin/bash
for idx in `seq 0 1000 20000`
do
start=$idx
end=$(($idx+1000))
# echo $start
	nohup /home/gjq/.conda/envs/android/bin/python -u apkPreprocess/script_preprocess/CG1_generateCG_multithread.py --start=$start --end=$end > apkPreprocess/process_Log/generateCG/92CG1_$start.log 2>&1 &
done
