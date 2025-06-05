import os
import subprocess
import time

if __name__ == "__main__":
    ApkRootName="apkPreprocess/TrainData/data2018/ApkTrainSet"
    cgRootName="apkPreprocess/TrainData/data2018/ApkPreProcessRes"
    for category in os.listdir(ApkRootName):
        pathName=os.path.join(ApkRootName,category)
        # print(pathName)
        for fileName in os.listdir(pathName):
            apkFilePath=os.path.join(pathName,fileName)
            apkName='.'.join(fileName.split('.')[:-1])
            CGpath=os.path.join(cgRootName,category,apkName)
            print(apkFilePath)
            print(CGpath)
            # apkFilePath="FlowDroid/javaProject/demo/errorApk/com.ashberrysoft.leadertask.apk"
            # CGpath="FlowDroid"
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

