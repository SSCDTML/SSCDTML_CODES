import os
import subprocess
import cv2
import scipy.io as sio

dir_act = os.getcwd()
name_folder = r'path_to_matlab_executable_file'
img_path = r'path_to_FFDM_image'
full_path = os.path.join(dir_act, name_folder)

output = subprocess.check_output([full_path, img_path])
output = output.decode('utf-8')

name = 'segmentacion.png'

new_image_path = os.path.join(dir_act, name)
im = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

path_pd = os.path.join(dir_act, 'PD.mat')
PD = sio.loadmat(path_pd)
pd = PD['PD_clustering']
pd = float(pd)

pd_value = pd
path_im = img_path
pd = pd*100
path_seg = os.path.join(dir_act, "segmentacion.png")
