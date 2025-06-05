#!/bin/bash
for idx in `seq 0 1000 20000`
do
start=$idx
end=$(($idx+1000))
# echo $start
	nohup /home/gjq/.conda/envs/android/bin/python -u apkPreprocess/script_preprocess/opcode2_smali2opcodeSeq.py --start=$start --end=$end > apkPreprocess/process_Log/Opcode2_Log/826opcode2_$start.log 2>&1 &
done
