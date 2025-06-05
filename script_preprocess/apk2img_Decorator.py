import os
import numpy as np
from androguard.core.bytecodes.apk import APK
from PIL import Image
import logging
import time
from shutil import move
# fileName=apkName.type  +  前面是pathName ==> fn 
# 换年份debug一下except
allcnt=0
badcnt=0
cnt32=0
cnt64=0

def saveImage(apkName:str,image:Image,file_type:str,category):
    # pathName = "apkPreprocess/TrainData/data2021/ApkPreProcessResTest"
    pathName = "apkPreprocess/TrainData/data_Reflection/ApkPreProcessRes"
    dir=os.path.join(pathName,category,apkName)
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_type=file_type[1:]
    fileName= f"{file_type}.png"
    print(os.path.join(dir,fileName))
    #print(os.path.join(dir,apkName))
    image.save(os.path.join(dir,fileName))

def get_bytes(apk: APK, file_type: str,category) -> bytes:
    assert file_type in {".dex", ".so", ".xml"}
    for f in apk.get_files():
        if f.endswith(file_type):
            if file_type==".dex":
                dexFile=apk.get_file(f)

                endian_tag=int.from_bytes(dexFile[40:44],"little")

                dexHeader=dexFile[0:112]

                dataSize=int.from_bytes(dexFile[104:108],"little")
                dataOff=int.from_bytes(dexFile[108:112],"little")
                dexData=dexFile[dataOff:dataOff+dataSize]

                dexRet=dexData

                if endian_tag!=0x12345678:
                    big=int.from_bytes(dexFile[40:44],"big")
                    print('error,endian_tag_little:%#x'%endian_tag)
                    print('error,endian_tag_big:%#x'%big)

                yield dexRet

            elif file_type==".xml":
                xmlFile=apk.get_file(f)

                xmlFileType=int.from_bytes(xmlFile[0:2],"little")
                fileHeaderSize=int.from_bytes(xmlFile[2:4],"little")
                fileSize=int.from_bytes(xmlFile[4:8],"little")

                if xmlFileType!=0x03 or fileHeaderSize!=0x08:
                    yield xmlFile

                else:
                    xmlRet=bytes()
                    curChunkStart=8
                    nextChunkStart=8
                    while nextChunkStart<fileSize:
                        curChunkStart=nextChunkStart
                        curChunkType=int.from_bytes(xmlFile[curChunkStart:curChunkStart+2],"little")
                        curChunkSize=int.from_bytes(xmlFile[curChunkStart+4:curChunkStart+8],"little")
                        if curChunkType==0x0001 or curChunkType==0x0102 or curChunkType==0x0103:
                            curChunk=xmlFile[curChunkStart:curChunkStart+curChunkSize]
                            xmlRet+=curChunk
                        nextChunkStart=curChunkStart+curChunkSize
                    yield xmlRet

            elif file_type==".so":
                elfFile=apk.get_file(f)
                elfFileType=int.from_bytes(elfFile[0:4],"big")
                if elfFileType!=0x7f454c46:
                    print('elfFileType:%#x'%elfFileType)                   
                    yield elfFile
                else:
                    elfRet=elfFile
                    elf32or64=elfFile[4]
                    if elf32or64==0x01: 
                        global cnt32
                        cnt32+=1
                        programHeaderOff=int.from_bytes(elfFile[28:32],"little")
                        sectionHeaderOff=int.from_bytes(elfFile[32:36],"little")
                        elfHeaderSize=int.from_bytes(elfFile[40:41],"little")
                        phSize=int.from_bytes(elfFile[42:43],"little")
                        phNum=int.from_bytes(elfFile[44:45],"little")
                        shSize=int.from_bytes(elfFile[46:47],"little")
                        shNum=int.from_bytes(elfFile[48:49],"little")
                        if programHeaderOff!=elfHeaderSize:
                            print(programHeaderOff)
                            print(elfHeaderSize)
                        # elfRet=elfFile[0:programHeaderOff]+elfFile[programHeaderOff+phSize*phNum:sectionHeaderOff]+elfFile[sectionHeaderOff+shSize*shNum:]
                        elfRet=elfFile[programHeaderOff+phSize*phNum:sectionHeaderOff]+elfFile[sectionHeaderOff+shSize*shNum:]

                        # elfRet[programHeaderOff:programHeaderOff+phSize*phNum]=[]
                        # elfRet[sectionHeaderOff:sectionHeaderOff+shSize*shNum]=[]

                    # 64                    
                    elif elf32or64==0x02:
                        global cnt64
                        cnt64+=1
                        programHeaderOff=int.from_bytes(elfFile[32:40],"little")
                        sectionHeaderOff=int.from_bytes(elfFile[40:48],"little")
                        elfHeaderSize=int.from_bytes(elfFile[52:53],"little")
                        phSize=int.from_bytes(elfFile[54:55],"little")
                        phNum=int.from_bytes(elfFile[56:57],"little")
                        shSize=int.from_bytes(elfFile[58:59],"little")
                        shNum=int.from_bytes(elfFile[60:61],"little")
                        if programHeaderOff!=elfHeaderSize:
                            print(programHeaderOff)
                            print(elfHeaderSize)
                        # elfRet=elfFile[0:programHeaderOff]+elfFile[programHeaderOff+phSize*phNum:sectionHeaderOff]+elfFile[sectionHeaderOff+shSize*shNum:]
                        elfRet=elfFile[programHeaderOff+phSize*phNum:sectionHeaderOff]+elfFile[sectionHeaderOff+shSize*shNum:]

                        # elfRet[programHeaderOff:programHeaderOff+phSize*phNum]=[]
                        # elfRet[sectionHeaderOff:sectionHeaderOff+shSize*shNum]=[]
                    else:
                        print("error!")
                    yield elfRet

def generate_png(apk: APK, apkName: str, file_type: str,category):
    assert file_type in {".dex", ".so", ".xml"}
    stream = bytes()
    for s in get_bytes(apk, file_type,category):
        stream += s
    current_len = len(stream)

    if current_len<224:
        stream=stream.ljust(224*224,b'\x00')
        current_len = len(stream)
    
    image = Image.frombytes(mode='L', size=(224, (int)(current_len/224)), data=stream) #width=224??
    image = image.resize((224,224), resample=Image.Resampling.BILINEAR)                 #BILINEAR??
    
    saveImage(apkName,image,file_type,category)

    return image

def generate_color_image(apk, apkName,category):
    dex_img = generate_png(apk, apkName, '.dex',category)
    xml_img = generate_png(apk, apkName, '.xml',category)
    so_img  = generate_png(apk, apkName,'.so' ,category)
    # if len(so_img)==0:
    #     so_img = np.zeros(dex_img.size)
    dex_img, so_img, xml_img = np.array(dex_img), np.array(so_img), np.array(xml_img)
    H, W = dex_img.shape
    print(H,W)
    image = np.zeros((H, W, 3))
    image[:, :, 0] = dex_img
    image[:, :, 1] = so_img
    image[:, :, 2] = xml_img
    color_image = Image.fromarray(image.astype(np.uint8))

    saveImage(apkName,color_image,'.color',category)
    

if __name__ == "__main__":
    cnt=0
    # rootName="apkPreprocess/TrainData/data2021/ApkTrainSetTest"
    rootName="apkPreprocess/TrainData/data_Reflection/ApkTrainSet"
    for category in os.listdir(rootName):
        if category == "benign":
            pathName=os.path.join(rootName,category)
            # print(pathName)
            for fileName in os.listdir(pathName):
                cnt+=1
                print("cnt:",cnt,time.ctime())
                fn=os.path.join(pathName,fileName)
                try:
                    apk = APK(fn)
                    apkName='.'.join(fileName.split('.')[:-1])
                    generate_color_image(apk, apkName,category)
                except:
                    print("Error****",fn)
