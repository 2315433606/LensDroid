import os
import shutil

if __name__ == "__main__":
    ApkRootName="apkPreprocess/TrainData/data2018/ApkTrainSet"
    for category in os.listdir(ApkRootName):
        pathName=os.path.join(ApkRootName,category)
        print(pathName)
        print(len(os.listdir(pathName)))
        
        for fileName in os.listdir(pathName):
            apkFilePath=os.path.join(pathName,fileName)
            name = '.'.join(fileName.split('.')[:-1])
            print(apkFilePath)
            print(name)
            if fileName.endswith('.zip'):
                newPath=os.path.join(pathName,name+'.apk')
                os.rename(apkFilePath , newPath)
            elif fileName.endswith('.apk'):
                continue
            else:
                os.remove(apkFilePath)