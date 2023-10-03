## Automatic Dense Tissue Segmentation in FFDM Based on Fully Convolutional Network and Intensity-Based Clustering (HYBRID)

This algorithm performs a segmentation of dense tissue using U-Net convolutional neural network. This segmentation is improved using k-means clustering. (_doi: 10.1109/ColCACI56938.2022.9905248._)

In this folder you can find the matlab file we used for the implementation of HYBRID algorithm and the python file used for using an executable of this matlab code in python.

You just need to enter the path of the dcm FFDM image and you obtain the breast density segmentation and percentage.

This code was provided by the authors. If you need any specific function (such as **'kmeansfgt'**), you could contact them. 

**'segBreast'**, **'ffdmRead'**, **'getinfo'** and **'semanticseg'** functions are available at OpenBreast.

OpenBreast repository available at: https://github.com/spertuz/openbreast
