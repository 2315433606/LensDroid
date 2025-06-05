import os
from re import M
import sys
from simplify_opcodeMap import simplify_opcodeMap
import time
import argparse
# fileName=apkName.type  +  前面是pathName ==> fn 
#在apkPreprocess/TrainData/data2021/ApkPreProcessResTest/benign下的文件中生成~.opcodeSqe.txt


def createOpcodeSeq(fileName:str,apkName:str,category):
    # from ApkSmaliFiles/malware/1 to dir:ApkPreProcessRes/malware/1/1.opcodeseq.txt
    pathName="apkPreprocess/TrainData/data_Reflection/ApkPreProcessRes"
    dir=os.path.join(pathName,category,apkName)
    if not os.path.exists(dir):
        print("not exist!")
        os.makedirs(dir)
        # print(dir,"not exist!")

    print(fileName,dir)

    dalvikOpcodes={}
    with open("apkPreprocess/script_preprocess/DalvikOpcodes.txt") as f:
        for line in f:
            (key,val)=line.split()
            dalvikOpcodes[key]=val
    # for key in dalvikOpcodes:
    #     print(key,dalvikOpcodes[key])
    opcodeSeqFileName=os.path.join(dir,apkName+".opcodeSeq.txt")
    with open(opcodeSeqFileName,'w') as f:
        for root, dirs, files in os.walk(fileName):
            for file in files:
                smaliFile = os.path.join(root, file)
                if(file.split('.')[-1]=="smali"):
                    opcodeSeq=smaliFileToOpcode(smaliFile, simplify_opcodeMap)
                    # opcodeSeq=smaliFileToOpcode(smaliFile, dalvikOpcodes)
                    # print(smaliFile)
                    # print(opcodeSeq)
                    f.write(opcodeSeq)


def smaliFileToOpcode(smaliFile:str,opcodeMap):
    opcodeSeq=''
    with open(smaliFile,'r') as f:
        msg=f.read()
        for i,part in enumerate(msg.split(".method")):
            # print(i,part)
            add_newline=False
            if i!=0:
                methodPart=part.split(".end method")[0]
                methodLines=methodPart.strip().split('\n')
                for line in methodLines:
                    line=line.strip()
                    if line and not line.startswith('.') and not line.startswith('#'):
                        opcode=line.split()[0]
                        if opcode in opcodeMap:
                            opcodeSeq+=opcodeMap[opcode]
                            add_newline=True
                if(add_newline):
                    # opcodeSeq += '\n'
                    opcodeSeq += '|'
    return opcodeSeq

if __name__ == "__main__":
    rootName="apkPreprocess/TrainData/data_Reflection/ApkSmaliFiles"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int,help="the start Index of apk sets")
    parser.add_argument('--end', type=int,help="the end Index of apk sets")
    args=parser.parse_args()
    cnt=0
    
    for category in os.listdir(rootName):
        if category=="malware":
            #pathName="apkPreprocess/TrainData/data2021/ApkSmaliFilesTest/benign"
            pathName=os.path.join(rootName,category)

            for fileName in os.listdir(pathName):
                cnt+=1
                if cnt<args.start or cnt>=args.end:
                    continue
                print(cnt,time.ctime())

                #fn="apkPreprocess/TrainData/data2021/ApkSmaliFilesTest/benign/a1.recommended.by.staff.reader.il1"
                fn=os.path.join(pathName,fileName)

                #apkName=
                #erro
                #apkName='.'.join(fileName.split('.')[:-1])

                apkName=fileName
                # print(fn,apkName,category)
                createOpcodeSeq(fn,apkName,category)
