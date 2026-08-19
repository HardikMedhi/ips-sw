import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter,medfilt,find_peaks,peak_widths
from datetime import datetime,timezone
from scipy import interpolate
from scipy.ndimage import median_filter

def median_abs_deviation(data,axis=None):
    '''
    Calculate median absolute deviation
    '''
    m=np.nanmedian(data, axis=axis, keepdims=True)
    abs_dev=np.abs(data-m)
    mad=np.nanmedian(abs_dev,axis=axis)
    return mad

def snr(data,mad_based=True):
    if mad_based:
        median=np.nanmedian(data)
        mad=median_abs_deviation(data)
        snr=0.6745*((data-median)/mad)
        return snr
    else:
        mean=np.nanmean(data)
        rms=np.abs(data**2)
        snr=mean/rms
        return snr
    
def time_rfi_filter(data,window_size=10,threshold=[5],axes=0):
    for thres in threshold:
        da=median_filter(data,size=window_size,axes=axes)
        zs = snr(da - data)
        mask = np.where(np.abs(zs) > thres)[0]
    return mask

data = pd.read_csv("0850-206_10chan.csv")
dat=data.to_numpy()
mask=time_rfi_filter(dat[:,0],window_size=500)
plt.plot(dat[:,0])
plt.scatter(mask,dat[mask,0],color='red')
plt.show()