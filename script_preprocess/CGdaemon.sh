#!/bin/bash
while true
do
	res=`ps -ef|grep jar|awk '$3==1 {print $2}'| wc -l`
	if [ $res -eq 0 ]
	then
        echo "$res"
    else
        cmd=`ps -ef|grep jar|awk '$3==1 {print $2}'| xargs kill -9`
		$cmd
	fi
	sleep 30s
done