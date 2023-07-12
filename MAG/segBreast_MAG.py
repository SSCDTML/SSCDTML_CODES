import numpy as np
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter
import cv2
from scipy import interpolate
from skimage import color, measure
from matplotlib import pyplot as plt
from sys import flags
import warnings
from scipy import signal
from scipy.optimize import curve_fit
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_fill_holes


# --------------------------------------------
# segChest function:
def segChest(im, contour):

    # Pre - process image:
    imc = cv2.dilate(im, disk(8))
    imc = gaussian_filter(imc, 1,)

    #### Detect pectoral line using Hough transform ####
    # Find edges:
    edge_map = cv2.Canny((imc*255).astype(np.uint8), 0.01*255, 0.06*255)
    edge_map = edge_map > 0

    # Crop edge map according to contour:
    # ROI2:
    ymax = np.round(0.6*np.max(contour['y']))
    xmax = np.round(np.min(contour['x'][contour['y'] < ymax]))
    edge_map = edge_map[0:int(ymax), 0:int(xmax)]

    # remove lower diagonal:
    m, n = edge_map.shape
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    y_ref = m - (m-1)*(x-1)/(n-1)
    edge_map[y > y_ref] = 0

    # Find edges coordinates:
    x, y = np.nonzero(edge_map)
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)

    # Quantize parameters space:
    N = 128  # Number of quantization points
    rho_max = np.min(edge_map.shape)
    rho_min = 1
    theta_min = 20*np.pi/180
    theta_max = 45*np.pi/180

    # Compute accumulation array A
    theta = np.expand_dims(np.linspace(theta_min, theta_max, N), -1).T
    rho = np.expand_dims(np.linspace(rho_min, rho_max, N), -1).T
    rho_k = x*np.cos(theta) + y*np.sin(theta)
    A = histc(rho_k, np.squeeze(rho))
    A = A[0]

    # Find maximum in accumulation array:
    imax = np.argmax(np.reshape(A, -1))
    i, j = ind2sub(A.shape, imax)

    # Get rect
    T = np.squeeze(theta)[j]
    R = (rho_max - rho_min)*(i - 1)/(N-1) + rho_min
    b = R/np.sin(T)
    m = -np.cos(T)/np.sin(T)

    # Get  mask
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    mask = np.ones(im.shape, dtype=bool)
    mask[y < b + m*x] = False
    cwall = {}
    cwall['m'] = m
    cwall['b'] = b

    return mask, cwall

def histc(x, bins):

    map_to_bins = np.digitize(x, bins)
    r = np.zeros((bins.shape[0], x.shape[1]))
    for j in range(x.shape[1]):
        for i in map_to_bins[:, j]:
            r[i-1, j] += 1
    return [r, map_to_bins]

def ind2sub(array_shape, ind):

    rows = int(ind.astype('int') / array_shape[1])
    cols = int(ind % array_shape[1])
    return (rows, cols)

# ---------------------------------------------------
# getcontour function:
def getcontour(mask, cflag):

    # Parameters #####################################
    K_th = 0.05  # Curvature threshold (higher is more strict)
                 # this works only if ismlo = true.
    elim = 8     # No of pixels to edge limit
    npts = 100   # number of contour points to return
    ##################################################

    # Find contour points:
    B, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    remov = np.logical_or(np.logical_or((B[0][:, 0, 0] <= elim), (B[0][:, 0, 1] <= elim)), (B[0][:, 0, 1] >= mask.shape[0] - elim))

    ys = B[0][np.logical_not(remov), 0, 1]
    ys = ys[::-1]

    xs = B[0][np.logical_not(remov), 0, 0]
    xs = xs[::-1]

    # sub - sample and smooth contour:
    n = len(xs)
    kernel = np.ones(5)/5
    xs = interpolate.interp1d(np.arange(n) + 1, xs, kind='cubic')(np.linspace(1, n, npts))
    xs = np.convolve(xs, kernel, mode='same')

    ys = interpolate.interp1d(np.arange(n) + 1, ys, kind='cubic')(np.linspace(1, n, npts))
    ys = np.convolve(ys, kernel, mode='same')

    # Crop contour by curvature analysis
    xc, yc, ycut = cropContour(xs, ys, K_th)

    contour = {}
    if not cflag:
        contour['x'] = xs
        contour['y'] = ys
    else:
        contour['x'] = xc
        contour['y'] = yc

    contour['ycut'] = ycut
    contour['size'] = mask.shape

    return contour

def cropContour(xs, ys, K_th, im=None):

    # compute curvature k:
    dx1 = np.diff(xs)      #dx/dt
    dx2 = np.diff(dx1)     #d2x/dt2
    dy1 = np.diff(ys)      #dy/dt
    dy2 = np.diff(dy1)     #d2y/dt2
    dx1 = dx1[1:]
    dy1 = dy1[1:]

    k = (dx1*dy2 - dy1*dx2)/((dx1**2 + dy1**2)**1.5)

    # Cut contour points with curvature
    # above the threshold K_th

    i = np.argmin(k)
    kmin = np.min(k)

    if np.abs(kmin) > K_th and xs[i] < 0.4*np.max(xs) and ys[i] > 0.5*np.max(ys):
        ycut = np.floor(ys[i])
        xs = xs[0:i+1]
        ys = ys[0:i+1]
    else:
        ycut = np.floor(np.max(ys))

    xc = xs
    yc = ys

    if im:
        plt.imshow(color.rgb2gray(im))

    return xc, yc, ycut

# --------------------------------------------
# ffdmForeground function:
def ffdmForeground(im, cflag=None):

    # Parameters: ####
    nbins = 1000 # Number wof bins for image histogram
    ##################

    if cflag == None or not cflag:
        cflag = False

    # find intensity threshold:
    warnings.simplefilter("ignore")
    xth = getLower(np.reshape(im, -1), nbins)
    warnings.simplefilter("default")

    # find mask:
    mask0 = im >= np.maximum(xth, 0)
 
    # remove artifacts and holes in the mask:
    mask = cleanMask(mask0)

    # compute contour:
    contour = getcontour(mask, cflag)
    contour['th'] = xth
    mask[int(np.round(np.max(contour['y']))):, :] = False

    return mask, contour

def funct(x, a, b, c):
  return a * np.exp(-((x-b)/c)**2)

def getLower(x, nbins):

    # Minimum and maximum intesities:
    x_min = np.min(x)
    x_max = np.max(x)

    # Relative frequency:
    xi = np.linspace(x_min, x_max, nbins)
    n = histc(np.expand_dims(x, -1), xi)
    n = np.squeeze(n[0])

    # Smooth histogram:
    n = np.convolve(n, signal.windows.gaussian(25, std=4.8), mode='same')
    n = n/np.max(n)

    # find threshold by fitting a gaussian to the
    # histogram peak(s) located below the mean.
    xsup = np.minimum(np.mean(x), np.maximum(np.percentile(x, 30), 0.2))
    ipeaks, _ = signal.find_peaks(n*(xi <= xsup), height=0.35)

    if len(ipeaks) == 1:
        # only one peak
        select = np.logical_and(n > 0.35, xi < xsup)
        f, _ = curve_fit(funct, xi[select], n[select])
        xth = f[1] + np.sqrt(f[2]**2 * (np.log(f[0]) - np.log(0.05)))

    elif len(ipeaks) > 1:   # two peaks
        # find minimum between peaks
        n_min = np.min(n[np.min(ipeaks):np.max(ipeaks)])
        i_min = np.argwhere(n == n_min)
        i_min = i_min[0]

        # adjust second peak
        select = xi >= xi[i_min] and n > 0.35 and xi < xsup
        f, _ = curve_fit(funct, xi(select), n(select))
        xth = f[1] + np.sqrt(f[2]**2 * (np.log(f[0]) - np.log(0.05)))

    elif not ipeaks:
        n_max = np.max(n)
        i_th = np.argwhere(n < 0.05*n_max)
        xth = xi[i_th[0]]

    return xth

def cleanMask(mask0):

    # remove first and last row
    mask0[0, :] = False
    mask0[-1, :] = False
    kernel = np.ones((5, 5), np.uint8)
    mask0 = cv2.erode(1*mask0.astype(np.uint8), kernel, iterations=1)

    # keep biggest region
    cc = measure.label(mask0)
    stats = measure.regionprops(cc)
    idx = 1 + np.argmax([i.area for i in stats])

    mask0 = (cc == idx)
    
    # remove spurious holes:
    mask0 = cv2.dilate(mask0.astype(np.uint8), kernel)
    mask0 = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel)

    h, w = mask0.shape[:2]
    mask1 = np.zeros((h + 2, w + 2), np.uint8)

    mask = cv2.threshold(mask0, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        cv2.fillPoly(mask, contours, 255)
    
    return mask0 
 
def histc(x, bins):
    map_to_bins = np.digitize(x, bins)
    r = np.zeros((bins.shape[0], x.shape[1]))
    for j in range(x.shape[1]):
        for i in map_to_bins[:, j]:
            r[i-1, j] += 1
        return [r, map_to_bins]


# --------------------------------------------------
## segBreast function:

def segBreast(im, ismlo, isffdm=None):

    # By default, assume that input is FFDM
    if isffdm == None:
        isffdm = True

    # flip image if necessary
    
    isflipped = isright(im)

    # Breast boundary detection:
    if isffdm:
        mask, contour = ffdmForeground(im, ismlo)

    else:
        mask, contour = sfmForeground(im, ismlo)

    #contour['flip'] = isflipped

    # Breast chest wall detection:
    if ismlo:
        cmask, cwall = segChest(im, contour)
    else:
        cmask = np.ones(mask.shape, dtype=bool)
        cwall = {}
        cwall['m'] = mask.shape[0]
        cwall['b'] = 0
    
    mask = np.logical_and(mask, cmask)
    mask, contour = ffdmForeground(im, ismlo)

    # flip back mask if necessary
    
    if isflipped:
        mask = np.fliplr(mask)
    mask, contour = ffdmForeground(im, ismlo)
    
    return mask, contour, cwall

def isright(im):
    im = np.where(im <= 0.95*np.max(im), im, 0)
    s = np.sum(im, axis=1)
    n = np.round(0.5*im.shape[1])

    flag = np.sum(s[0:int(n)]) < np.sum(s[int(n):])
    return flag