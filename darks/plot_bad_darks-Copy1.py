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


bad_ch4_pfx_list=open('noisy_ch4_id').readlines()
for ipfx,pfx in enumerate(bad_ch4_pfx_list):
    bad_ch4_pfx_list[ipfx] = pfx.strip('\n')
    print(f'Dark ID {ipfx}: {pfx.strip('\n'}')
    os.makedirs(f'figs/{pfx.strip('\n')}',exist_ok=True)

bad_o2_pfx_list=open('noisy_o2_id').readlines()
for ipfx,pfx in enumerate(bad_o2_pfx_list):
    bad_o2_pfx_list[ipfx] = pfx.strip('\n')
    print(f'Dark ID {ipfx}: {pfx.strip('\n')}')
    os.makedirs(f'figs/{pfx.strip('\n')}',exist_ok=True)


bad_o2_files = {}
bad_ch4_files = {}
for pfx in bad_o2_pfx_list:
    o2_inds = np.where([str(s).split('/')[3] == f'{pfx}' for s in o2_files])[0].flatten()
    if len(o2_inds) == 0:
        print('Warning: {pfx} had no matching O2 files')
        continue
    bad_o2_files[pfx] = o2_files[o2_inds]

for pfx in bad_ch4_pfx_list:
    ch4_inds = np.where([str(s).split('/')[3] == f'{pfx}' for s in ch4_files])[0].flatten()
    if len(o2_inds) == 0:
        print('Warning: {pfx} had no matching CH4 files')
        continue
    bad_ch4_files[pfx] = ch4_files[ch4_inds]

# Plot all dark images for a given collect
for ip,pfx in enumerate(sorted(list(bad_o2_files.keys()))):
    print(pfx)
    os.makedirs(os.path.dirname(bad_o2_files[pfx][-1]),exist_ok=True)
    with Dataset(bad_o2_files[pfx][-1] as f:
        dn = f['Frame/PixelData'][:]
        n_rows = dn.shape[0]//4+1
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
        fig.savefig(f'figs/frames/{pfx}/{pfx}_o2_darks.png')
        plt.close('all')
    del dn