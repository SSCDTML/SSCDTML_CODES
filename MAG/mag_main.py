from mseg_MAG import *
from pprocess_MAG import *
from segBreast_MAG import *

from PIL import Image
import cv2
import os
import numpy as np

path_im = 'image_path'
im = cv2.imread(path_im, cv2.IMREAD_GRAYSCALE)

mask,_,_ = segBreast(im,False,True) 
mask= (mask* 255).astype(np.uint8)

mask = Image.fromarray(mask)
mask.save(os.path.join("mask.png"))
            
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE) 
seg,pd,_ = mseg(im, mask, 0.4)

dir_act = os.getcwd()



seg= (seg* 255).astype(np.uint8)
seg = Image.fromarray(seg)
seg.save(os.path.join("segmentacion.png"))

path_seg = os.path.join(dir_act, "segmentacion.png")