# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:27:16 2018

@author: Administrator
"""

from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=100,height=100):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("E:/projects/2019.3.12/tumu_1/test_1\\*.jpg"):
    convertjpg(jpgfile,"E:/projects/2019.3.12/tumu_1/test_100")
