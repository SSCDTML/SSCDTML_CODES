from data import *
from modelUnetResnet50 import *

import os

vid = "v001_3"
img_folder = r'path_to_folder_of_images_to_be_segmented'
path_save = r'path_to_save_segmentation'
full_path_unet  = r'path_to_trained_CNN'
original_name = os.path.basename(img_folder)
original_name = os.path.splitext(original_name)[0]
model = build_resnet50_unet()
testGene = testGenerator(img_folder)
model.load_weights(full_path_unet)
results = model.predict_generator(testGene,1,verbose=0)
path_multi = saveResult(original_name, path_save,results, vid[:4])