from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import glob,os,pdb
import concurrent.futures
import warnings
import shutil
import scipy

def calculate_bad_pixels(mean_dark,std_dark,bp=None):
    dead = np.zeros(mean_dark.shape)
    hot = dead.copy()
    cold = dead.copy()
    high_std = dead.copy()
    if bp is None:
        bp = np.zeros(dead.shape)
    dead[np.logical_and(std_dark < 1,bp<1)] = 1  # Dead Pixels
    dead[std_dark.mask] = 0
    
    cold[np.logical_and(mean_dark < 1000, std_dark < 1,bp<1)] = 1  # COLD Pixels
    cold[std_dark.mask] = 0
    
    hot[np.logical_and(mean_dark > 3000, std_dark < 1,bp<1)] = 1  # HOT Pixels
    hot[std_dark.mask] = 0
    
    high_std[np.logical_and(std_dark > 5 * np.nanmean(std_dark),bp<1)] = 1
    high_std[std_dark.mask] = 0

    return {'dead':dead[:],'cold':cold[:],'hot':hot[:],'noisy':high_std[:]}

def denoise_replace_mask(dn=None,thresh=1.5,pf_bp=None,filter='median'):
    bp_c = calculate_bad_pixels(dn.mean(0),dn.std(0),bp=pf_bp)
    y,x = np.where(bp_c['noisy'] == 1)

    new_mask = np.zeros(dn.shape)
    if filter == 'mean':
        new_mask[:,y,x] = np.array([np.abs(dn[:,yy,xx] - np.nanmean(dn[:,yy,xx],0)) > thresh*np.nanstd(dn[:,yy,xx],0) for xx,yy in zip(x,y)]).T
    elif filter == 'median':
        new_mask[:,y,x] = np.array([np.abs(dn[:,yy,xx] - np.nanmean(dn[:,yy,xx],0)) <= thresh*scipy.stats.iqr(dn[:,yy,xx]) for xx,yy in zip(x,y)]).T

    denoised_dark = dn.copy()
    denoised_dark.mask[:,y,x] = np.logical_or(denoised_dark.mask[:,y,x],new_mask[:,y,x])

    new_bp = calculate_bad_pixels(denoised_dark.mean(0),denoised_dark.std(0),bp=pf_bp)
    return bp_c,new_bp,denoised_dark

def outlier_filter_replace_values(dn=None,thresh=1.5,pf_bp=None,filter='mean'):
    bp_c = calculate_bad_pixels(dn.mean(0),dn.std(0),bp=pf_bp)
    y,x = np.where(np.logical_or.reduce([bp_c['noisy'] == 1,bp_c['hot'] == 1,bp_c['cold']==1,bp_c['dead']==1]))

    new_vals = dn.copy()
    
    if filter == 'mean':
        try:
            for xx,yy in zip(x,y):
                bad_inds = np.where(dn[:,yy,xx] > np.nanmean(dn[:,yy,xx],0) + thresh*np.nanstd(dn[:,yy,xx],0))[0]
                good_inds = np.where(dn[:,yy,xx] <= np.nanmean(dn[:,yy,xx],0) + thresh*np.nanstd(dn[:,yy,xx],0))[0]
                new_vals[bad_inds,yy,xx] = np.nanmean(dn[good_inds,yy,xx],0)
        except:
            pdb.set_trace()
    elif filter == 'median':
        for xx,yy in zip(x,y):
            inds = np.where(dn[:,yy,xx] > np.nanmedian(dn[:,yy,xx],0) + thresh*scipy.stats.iqr(dn[:,yy,xx]))[0]
            new_vals[inds,yy,xx] = np.nan
            new_vals[inds,yy,xx] = np.nanmedian(new_vals[:,yy,xx],0)
            
    filter_dark = new_vals.copy()
    filter_dark.mask = dn.mask.copy()
    
    new_bp = calculate_bad_pixels(filter_dark.mean(0),filter_dark.std(0),bp=pf_bp)
    return bp_c,new_bp,filter_dark

def create_filtered_dark(ID=None,thresh=1.5,pf_bp=None,out_dir=None):
    #ID is the filename
    fname = os.path.basename(ID)
    new_fname = f'{out_dir}/{fname}'

    os.makedirs(f'{out_dir}',exist_ok=True)
    shutil.copy2(ID,new_fname)
    
    with Dataset(new_fname,'r+') as fid:
        dn = fid['Frame/PixelData'][:]
        bp_c,new_bp,new_dark = outlier_filter_replace_values(dn,thresh=thresh,pf_bp=pf_bp)
        fid['Frame/PixelData'][:] = new_dark.copy()
    return


def run_function_in_parallel(fun,args_list):
    """
                        to multithreading issues with rpy2
    Inputs:
        fun             Python function
    Returns:
                        consisting of the Exception.
    """
    result_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        # Submit tasks to be executed in parallel with named arguments
        future_to_args = {executor.submit(fun,**args): args for args in args_list}
        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_args):
            args_future = future_to_args[future]
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = future.result()
                    result_dict[args_future['ID']] = {
                            "result": result,
                            "warnings": [str(warn.message) for warn in w],
                            "error": None
                            }
                print(f"Function with arg ID {args_future['ID']} finished successfully!")
            except Exception as exc:
                print(f"Function with arg ID {args_future['ID']} generated an exception: {exc}")
                result_dict[args_future['ID']] = {
                            "result": None,
                            "warnings": [],
                            "error": f"Generated an exception: {exc}"
                            }
    return result_dict

if __name__ == "__main__":

    
    ch4_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_CH4_BadPixelMap_CH4_20250722.nc','r')['BadPixelMap'][:] 
    ch4_files = sorted(glob.glob('/mnt/share1/sean/original/*CH4*'))
    #ch4_files = [f.rstrip('\n') for f in open('ch4_dark_files').readlines()]
    #ch4_folders = [os.path.dirname(fi) for fi in ch4_files]
    ch4_args_list = [
                    {'ID':ch4_files[i],'pf_bp':ch4_bp,'out_dir':'/mnt/share1/sean/filtered'} for i in range(len(ch4_files))
                    ]
    ch4_out = run_function_in_parallel(create_filtered_dark,ch4_args_list)

    
    o2_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_O2_BadPixelMap_O2_20250722.nc','r')['BadPixelMap'][:] 
    o2_files = sorted(glob.glob('/mnt/share1/sean/original/*O2*'))
    #o2_files = [f.rstrip('\n') for f in open('o2_dark_files').readlines()]
    #o2_folders = [os.path.dirname(fi) for fi in o2_files]
    o2_args_list = [
                    {'ID':o2_files[i],'pf_bp':o2_bp,'out_dir':'/mnt/share1/sean/filtered'} for i in range(len(o2_files))
                    ]
    o2_out = run_function_in_parallel(create_filtered_dark,o2_args_list)
