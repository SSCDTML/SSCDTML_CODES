import numpy as np
from scipy.signal import gaussian
from pprocess_MAG import pprocess
import scipy.ndimage as sn
from scipy.signal import convolve

def mseg(im, mask, pixel_size):
    
    sigma = 0.8 #std of smoothing gaussian filter
    n = 25 #number of gray levels
    
    #Post-processing parameters
    
    skin_gap = 8 #skin gap in mm
    area_th = 16 #area threshold in mm^2
    
    #Smoothing filter
    
    length = 5 * sigma + 1
    h = gaussian(length, sigma)
    
    #Intensity levels
    
    imin = np.amin(im[mask>0])
    imax = np.amax(im[mask>0])
    ivalues = np.linspace(imin, imax, n)
    
    #Compute morphological area curve
    
    area = np.zeros((n,1))
    
    for k in range(n):
        
        seg = (im >= ivalues[k]) & mask
        area[k] = np.sum(seg)
    
    #Compute first morphological area gradient (MAG)
    diff = np.array([1, -1]);
    mag = convolve(np.reshape(area,[n,]),diff,'valid')

    # Smooth MAG to remove noise
    mag = np.convolve(mag, np.array(h), mode='same')

    # Minimize MAG
    i = np.argmin(mag)
    
    # Segment image
    seg = (im >= ivalues[i+1]) & mask
    
    # Post-process image
    seg = pprocess(seg, mask, skin_gap, area_th, pixel_size)
    seg = seg*1

    # Density percentage
    pd = np.sum(seg.flatten()*255) / np.sum(mask.flatten())*100
    return seg, pd, mask