import os
import numpy as np
from androguard.core.bytecodes.apk import APK
from PIL import Image
import time


cnt=0
year="data2018"

def saveImage(apkName:str,image:Image,file_type:str,category):
    pathName = "apkPreprocess/TrainData/{}/ApkPreProcessRes".format(year)
    dir=os.path.join(pathName,category,apkName)
    if not os.path.exists(dir):
        print("dirError****",fn)
        os._exit()
    file_type=file_type[1:]
    fileName= "{}{}.png".format(file_type,"Untreated")
    print(os.path.join(dir,fileName))
    image.save(os.path.join(dir,fileName))

def get_bytes(apk: APK, file_type: str) -> bytes:
    assert file_type in {".dex", ".so", ".xml"}
    for f in apk.get_files():
        if f.endswith(file_type):
            yield apk.get_file(f)

def generate_png(apk: APK, apkName: str, file_type: str,category):
    assert file_type in {".dex", ".so", ".xml"}
    stream = bytes()
    for s in get_bytes(apk, file_type):
        stream += s
    current_len = len(stream)
    print(current_len)
    if current_len<224:
        stream=stream.ljust(224*224,b'\x00')
        current_len = len(stream)
    image = Image.frombytes(mode='L', size=(224, (int)(current_len/224)), data=stream) #width=224??
    image = image.resize((224,224), resample=Image.Resampling.BILINEAR)                 #BILINEAR??
    return image

def generate_color_image(apk, apkName,category):
    dex_img = generate_png(apk, apkName, '.dex',category)
    xml_img = generate_png(apk, apkName, '.xml',category)
    so_img  = generate_png(apk, apkName, '.so',category)
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
    rootName="apkPreprocess/TrainData/{}/ApkTrainSet".format(year)
    for category in os.listdir(rootName):
        if(category=="benign"):
            pathName=os.path.join(rootName,category)
            for fileName in os.listdir(pathName):
                cnt+=1
                print("cnt:",cnt,time.ctime())
                fn=os.path.join(pathName,fileName)
                try:
                    apk = APK(fn)
                    apkName=""
                    if year=="Drebin":
                        apkName=fileName                        
                    else:
                        apkName='.'.join(fileName.split('.')[:-1])                     
                    generate_color_image(apk, apkName,category)
                except:
                    print("Error****",fn)
