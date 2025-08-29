# Use NCO to compute Mean/Std of all dark files
import numpy as np
from pathlib import Path
import glob,os,pdb
import subprocess
from netCDF4 import Dataset
import concurrent.futures
import warnings

def run_function_in_parallel(fun,args_list):
    """
    Description:        Generic function that runs any function over a set of CPUs -
                        note that this uses concurrent.futures.PoolProcessExecutor due
                        to multithreading issues with rpy2
    Inputs:
        fun             Python function
        args_list       List of dictionaries with keys set to match the input variable
                        argument names for the function fun. Note: each dictionary must have
                        a key called ID with unique value to be identifiable in the results dictionary.
    Returns:
        results_dict    Concatenated results of the function from collection of args_list
                        indexed by the ID key in the args_list list of dictionaries. Each entry
                        has a "result" entry that is the return value of the function, a "warnings" 
                        entry, and if the function does not conclude successfully, an error entry 
                        consisting of the Exception.
    """
    result_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        # Submit tasks to be executed in parallel with named arguments
        future_to_args = {executor.submit(fun, **args): args for args in args_list}

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

def calc_mean(file=None):
    subprocess.run(['ncra','-O','-v','PixelData',fi,f'mean_darks/{str(fi).split("/")[-1][:-3]}_mean.nc'])
    return

def calc_std(file=None):
    subprocess.run(["ncwa","-O","-v","PixelData",str(fi),"temp.nc"])
    subprocess.run(["ncbo","-O","-v","PixelData",str(fi),"temp.nc","temp.nc"])
    subprocess.run(["ncra","-O","-y","rmssdn","temp.nc",f"std_darks/{str(fi).split('/')[-1][:-3]}_std.nc"])
    os.remove("temp.nc")
    return
    
def calc_mean_std(ID=None,file=None):
    dn = Dataset(file,'r')['Frame/PixelData'][:]
    ts = Dataset(file,'r')['Frame/TimeStamp'][:][-1]
    n_frames = 0.
    for i_fm in range(dn.shape[0]):
        if dn[i_fm].mask.sum() > 0.9*dn.shape[1]*dn.shape[2]:
            continue
        n_frames += 1
    
    mean_dn = np.nanmean(dn,0)
    std_dn = np.nanstd(dn,0)
    out_dir = f'mean_darks/{os.path.dirname(file)}'
    os.makedirs(out_dir,exist_ok=True)
    out_fname = f'mean_darks/{file[:-3]}_mean.nc'
    
    if os.path.exists(out_fname): os.remove(out_fname)
    with Dataset(out_fname,'w') as fid:
        fid.createDimension('across_track',dn.shape[1])
        fid.createDimension('spectral_pixel',dn.shape[2])
        fid.timestamp=ts
        fid.framecount=n_frames
        v = fid.createVariable('mean_dark','f4',('across_track','spectral_pixel'),compression='zlib',least_significant_digit=1)
        v[:] = mean_dn[:]
        v = fid.createVariable('std_dark','f4',('across_track','spectral_pixel'),compression='zlib',least_significant_digit=1)
        v[:] = std_dn[:]
        v = fid.createVariable('total_masked','i2',('across_track','spectral_pixel'),compression='zlib')
        v[:] = dn.mask.sum(0)
        v = fid.createVariable('max_dn','i2',('across_track','spectral_pixel'),compression='zlib')
        v[:] = dn.max(0)
        v = fid.createVariable('min_dn','i2',('across_track','spectral_pixel'),compression='zlib')
        v[:] = dn.min(0)
        v = fid.createVariable('median_dn','i2',('across_track','spectral_pixel'),compression='zlib')
        v[:] = np.nanmedian(dn,0)
        #fid.close()

    #subprocess.run(['ncap2','-v','-O','-s','mean_dark=pack(mean_dark,1,0);std_dark=pack(std_dark)',out_fname,out_fname])
    
    return


if __name__ == "__main__":
    ch4_files = list(Path('./2024/').rglob('*CH4*.nc'))
    ch4_files.extend(list(Path('./2025/').rglob('*CH4*.nc')))
    ch4_files = sorted(ch4_files)
    
    o2_files = list(Path('./2024/').rglob('*O2*.nc'))
    o2_files.extend(list(Path('./2025/').rglob('*O2*.nc')))
    o2_files = sorted(o2_files)
    
    os.makedirs('mean_darks',exist_ok=True)
    o2_args_list = [{'ID':str(fi),'file':str(fi)} for fi in o2_files]
    o2_out = run_function_in_parallel(calc_mean_std,o2_args_list)
    ch4_args_list = [{'ID':str(fi),'file':str(fi)} for fi in ch4_files]
    ch4_out = run_function_in_parallel(calc_mean_std,ch4_args_list)
    
    #for ifi,fi in enumerate(o2_files):
    #    print(f'Calculating Mean O2 Dark #{ifi}/{len(o2_files)}: {str(fi).split("/")[-1]}')
    #    calc_mean_std(file=str(fi))
    #for ifi,fi in enumerate(ch4_files):
    #    print(f'Calculating Mean CH4 Dark #{ifi}/{len(ch4_files)}: {str(fi).split("/")[-1]}')
    #    calc_mean_std(file=str(fi))