import os
import subprocess
import time
import argparse
import sys

if __name__ == "__main__":


    ApkRootName="apkPreprocess/TrainData/data_Reflection/ApkTrainSet"
    cgRootName="apkPreprocess/TrainData/data_Reflection/ApkPreProcessRes"



    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int,help="the start Index of apk sets")
    parser.add_argument('--end', type=int,help="the end Index of apk sets")
    args=parser.parse_args()
    
    # if not args.start or not args.end:
    #     parser.print_help()
    #     sys.exit(0)

    cnt=0

    #查看ApkRootName下的目录
    for category in os.listdir(ApkRootName):
        if(category!="malware"):
            pathName=os.path.join(ApkRootName,category)
            # print(pathName)
            for fileName in os.listdir(pathName):
                cnt+=1
                if cnt<args.start or cnt>=args.end:
                    continue

                apkFilePath=os.path.join(pathName,fileName)
                apkName='.'.join(fileName.split('.')[:-1])
                CGpath=os.path.join(cgRootName,category,apkName)
                print(apkFilePath)
                print(CGpath)
                #zjl
                # 检查 CGpath 指向的目录是否存在
                if not os.path.exists(CGpath):
                    os.makedirs(CGpath)
                # apkFilePath="FlowDroid/javaProject/demo/errorApk/com.ashberrysoft.leadertask.apk"
                # CGpath="FlowDroid" androidMalware/androidMalware.jar
                
                try:
                    print(time.ctime())
                    subprocess.run('java -jar androidMalware.jar {} {}'.format(apkFilePath,CGpath),shell=True,timeout=300,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    # subprocess.Popen('java -jar androidMalware.jar {} {}'.format(apkFilePath,CGpath),shell=True,timeout=300,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                
                except subprocess.CalledProcessError as e1:
                    print(time.ctime())
                    print("apkError:",apkFilePath)
                except subprocess.TimeoutExpired as e2:
                    print(time.ctime())
                    print("TimeOut:",apkFilePath)

