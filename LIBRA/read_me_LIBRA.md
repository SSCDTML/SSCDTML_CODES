## Laboratory for Individualized Breast Radiodensity Assessment (LIBRA)

This algorithm applies an adaptative multiclass c-means algorithm to identify and partition breast tissue area into multiple regions. These clusters are then aggregated by a support-vector machine to a final dense tissue area segmentation. Finally, the ratio of the segmented absolute dense area to the total breast area is then used to obtain a measure of breast percent density (_doi: 10.1118/1.4736530_).

In this folder you can find the script in matlab used for implementing a segmentation using LIBRA. You can also find the python script employed for implementing the executable file from matlab in python.

Specific function such as **'getinfo'**, **'pdensity'**, and more are available at OpenBreast repository:
https://github.com/spertuz/openbreast/blob/master/demos/demo05.m
