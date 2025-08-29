# Plot bad pixel stats
from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import glob,os,pdb

def calculate_bad_pixels(dark):
    dead = np.zeros(dark.shape[1:])
    hot = dead.copy()
    cold = dead.copy()
    high_std = dead.copy()
    temp_dark_std = np.nanstd(dark,0)
    temp_dark_mean = np.nanmean(dark,0)
    dead[temp_dark_std < 1] = 1  # Dead Pixels
    cold[
        np.logical_and(temp_dark_mean < 1000, temp_dark_std < 1)
    ] = 1  # COLD Pixels
    hot[
        np.logical_and(temp_dark_mean > 3000, temp_dark_std < 1)
    ] = 1  # HOT Pixels
    high_std[temp_dark_std > 5 * np.nanmean(temp_dark_std)] = (
        1
    )
    return {'dead':dead[:],'cold':cold[:],'hot':hot[:],'high_std':high_std[:]}

os.chdir('/media/sata/methanesat/darks/')
ch4_files = list(Path('./2024/').rglob('*CH4*.nc'))
ch4_files.extend(list(Path('./2025/').rglob('*CH4*.nc')))
ch4_files = sorted(ch4_files)
    
o2_files = list(Path('./2024/').rglob('*O2*.nc'))
o2_files.extend(list(Path('./2025/').rglob('*O2*.nc')))
o2_files = sorted(o2_files)

ch4_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_CH4_BadPixelMap_CH4_20250722.nc','r')['BadPixelMap'][:] 
o2_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_O2_BadPixelMap_O2_20250722.nc','r')['BadPixelMap'][:] 

plt.close('all')
bad_o2_files = {}
bad_o2_file_list = open('noisy_o2_id','r').readlines()
for ifi,fi in enumerate(bad_o2_file_list):
    pfx = fi.split('/')[3]
    bad_o2_files[pfx] = fi.strip('\n')
for ip,pfx in enumerate(list(bad_o2_files.keys())):
    print(bad_o2_files[pfx])
    f = Dataset(bad_o2_files[pfx])
    dn = f['Frame/PixelData'][:]
    n_rows = max(((dn.shape[0])//4+1,1))
    fig,axs=plt.subplots(n_rows,4,figsize=(10,10))
    for i in range(dn.shape[0]):
        if dn.shape[0] < 4:
            ax = axs[i]
        else:
            ax = axs[i//4,i%4]
        g = ax.pcolormesh(dn[i],vmin=1000,vmax=1500); 
        plt.colorbar(g,ax=ax)
        ax.set_title(f'{i}')
    fig.tight_layout()
    fig.suptitle(f'{pfx} Dark Map')
    fname = '_'.join(bad_o2_files[pfx].split('/')[:-1])+bad_o2_files[pfx].split('/')[-1][:-3]+'_o2_darks.png'
    fig.savefig(f'figs/all_frames/{fname}')
    plt.close('all')
    del dn

plt.close('all')
bad_ch4_files = {}
bad_ch4_file_list = open('noisy_ch4_id','r').readlines()
for ifi,fi in enumerate(bad_ch4_file_list):
    pfx = fi.split('/')[3]
    bad_ch4_files[pfx] = fi.strip('\n')
for ip,pfx in enumerate(list(bad_ch4_files.keys())):
    print(pfx)
    f = Dataset(bad_ch4_files[pfx])
    dn = f['Frame/PixelData'][:]
    n_rows = max((dn.shape[0]//4+1,1))
    fig,axs=plt.subplots(n_rows,4,figsize=(10,10))
    for i in range(dn.shape[0]):
        if dn.shape[0] < 4:
            ax = axs[i]
        else:
            ax = axs[i//4,i%4]
        g = ax.pcolormesh(dn[i],vmin=1000,vmax=1500); 
        plt.colorbar(g,ax=ax)
        ax.set_title(f'{i}')
    fig.tight_layout()
    fig.suptitle(f'{pfx} Dark Map')
    fig.savefig(f'figs//all_frames/{bad_ch4_files[pfx][:-3]}__ch4_darks.png')
    plt.close('all')
    del dn