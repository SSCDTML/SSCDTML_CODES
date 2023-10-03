import os
import subprocess
import cv2
import scipy.io as sio
import numpy as np
from PIL import Image


dir_act = os.getcwd()
name_folder = r'path_to_matlab_executable_fil'
img_path = r'path_to_FFDM_image'
full_path = os.path.join(dir_act, name_folder)
output = subprocess.check_output([full_path, img_path])

output = output.decode('utf-8')
lines = output.splitlines()

name_PDfile = "PD.mat"
full_path_PD = os.path.join(dir_act, name_PDfile)
PD = sio.loadmat(full_path_PD)
PD = PD['PD']
PD = float(PD)
algorithm = 'LIBRA'
pd_value = PD
path_im = img_path

PD = PD*100
print("PD: ", PD)

name_maskDensefile = "maskDense.mat"
full_path_maskDense = os.path.join(dir_act, name_maskDensefile)
maskDense = sio.loadmat(full_path_maskDense)
maskDense= maskDense['maskDense']

maskDense = (np.array(maskDense) * 255).astype(np.uint8)
maskDense = Image.fromarray(maskDense)
maskDense = maskDense.resize((512, 512))
maskDense.save(os.path.join("segmentacion.png"))
                    