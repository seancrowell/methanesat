# Plot bad pixel stats
from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import glob,os,pdb,sys
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


def calculate_bad_pixels(mean_dark,std_dark,bp=None):
    dead = np.zeros(mean_dark.shape)
    hot = dead.copy()
    cold = dead.copy()
    high_std = dead.copy()
    if bp is None:
        bp = np.zeros(std_dark.shape)

    dead[
        np.logical_and(std_dark < 1,bp<1)
    ] = 1  # Dead Pixels
    dead[std_dark.mask] = 0
    
    cold[
        np.logical_and(mean_dark < 1000, std_dark < 1,bp<1)
    ] = 1  # COLD Pixels
    cold[std_dark.mask] = 0
    
    hot[
        np.logical_and(mean_dark > 3000, std_dark < 1,bp<1)
    ] = 1  # HOT Pixels
    hot[std_dark.mask] = 0

    high_std[
        np.logical_and(std_dark > 5 * np.nanmean(std_dark),bp<1)
    ] = 1
    high_std[std_dark.mask] = 0

    return {'dead':dead[:],'cold':cold[:],'hot':hot[:],'noisy':high_std[:]}

def plot_dark_panel(file=None,ID=None,fpa='O2',pf_bp=None,savedir='figs/mean_darks/'):
    f = Dataset(file)
    
    collect_id = str(file).split('/')[-1].split('_')[5]
    os.makedirs(f'{savedir}',exist_ok=True)
    
    dn = f['Frame/PixelData'][:]
    mean_dark = np.nanmean(dn,0)
    std_dark = np.nanstd(dn,0)
    bp = calculate_bad_pixels(mean_dark,std_dark,bp=pf_bp)

    n_rows = 3
    fig,axs=plt.subplots(n_rows,2,figsize=(10,10))
    #Mean Dark
    ax = axs[0,0]
    g = ax.pcolormesh(mean_dark,vmin=1000,vmax=1500); 
    plt.colorbar(g,ax=ax)    
    ax.set_title(f'{collect_id} {fpa} Mean Dark')
    #Std Dark
    ax = axs[0,1]
    g = ax.pcolormesh(std_dark,vmin=0,vmax=5); 
    plt.colorbar(g,ax=ax)
    ax.set_title(f'{collect_id} {fpa} Std Dark')

    #Dead Pixels
    ax = axs[1,0]
    #g = ax.pcolormesh(bp['dead'],vmin=0,vmax=1);
    y,x = np.where(bp['dead'])
    ax.scatter(x,y,c='tab:brown',s=3)
    ax.set_title(f'{collect_id} {fpa} Dead Pixels (N={bp["dead"].sum()})')
    #Cold Pixels
    ax = axs[1,1]
    #g = ax.pcolormesh(bp['cold'],vmin=0,vmax=1); 
    y,x = np.where(bp['cold'])
    ax.scatter(x,y,c='tab:blue',s=3)
    ax.set_title(f'{collect_id} {fpa} Cold Pixels (N={bp["cold"].sum()})')
    #Hot Pixels
    ax = axs[2,0]
    #g = ax.pcolormesh(bp['hot'],vmin=0,vmax=1); 
    y,x = np.where(bp['hot'])
    ax.scatter(x,y,c='tab:red',s=3)
    ax.set_title(f'{collect_id} {fpa} Hot Pixels (N={bp["hot"].sum()})')
    #High Std Pixels
    ax = axs[2,1]
    #g = ax.pcolormesh(bp['high_std'],vmin=0,vmax=1); 
    y,x = np.where(bp['noisy'])
    ax.scatter(x,y,c='tab:orange',s=3)
    ax.set_title(f'{collect_id} {fpa} Noisy Pixels (N={bp["noisy"].sum()})')

    fig.tight_layout()
    fig.suptitle(f'{collect_id} {fpa}')
    fname = '_'.join(file.split('/')[:-1])+f'_{fpa}_bp_panel.png'
    fig.savefig(f'{savedir}/{fname}')
    plt.close('all')
    return

def plot_all_frames(ID=None,file=None,fpa='O2'):
    os.makedirs(f'figs/all_frames/{os.path.dirname(file)}',exist_ok=True)
    with Dataset(file,'r') as f:
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
        fig.suptitle(f'{pfx} {fpa} Dark Frames')
        fig.savefig(f'figs/all_frames/{os.path.dirname(file)}/{pfx}_{fpa}_darks.png')
        plt.close('all')
        del dn
    return

if __name__ == "__main__":

    os.chdir('/media/sata/methanesat/darks/')
    try:
        dark_dir = sys.argv[2]
        fig_save_dir = sys.argv[3]
        plot_type = sys.argv[1]
    except:
        print('usage: python plot_darks.py plot_type dark_dir fig_save_dir')
        sys.exit()

    if plot_type == 'mean':
        ch4_files = np.array(sorted(list(Path(dark_dir).rglob('*_CH4_*.nc'))))
        o2_files = np.array(sorted(list(Path(dark_dir).rglob('*_O2_*.nc'))))
        ch4_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_CH4_BadPixelMap_CH4_20250722.nc','r')['BadPixelMap'][:] 
        o2_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_O2_BadPixelMap_O2_20250722.nc','r')['BadPixelMap'][:] 
        ch4_args = [{'ID':str(fi),'file':str(fi),'fpa':'CH4','pf_bp':ch4_bp,'savedir':fig_save_dir+'/ch4/'} for fi in ch4_files]
        #plot_dark_panel(**ch4_args[0])
        run_function_in_parallel(plot_dark_panel,ch4_args)
        o2_args = [{'ID':str(fi),'file':str(fi),'fpa':'O2','pf_bp':o2_bp,'savedir':fig_save_dir+'/o2/'} for fi in o2_files]
        #plot_dark_panel(**o2_args[0])
        run_function_in_parallel(plot_dark_panel,o2_args)
        
    elif plot_type == 'frames':
        all_ch4_files = list(Path('./2024/').rglob('*CH4*.nc'))
        all_ch4_files.extend(list(Path('./2025/').rglob('*CH4*.nc')))
        all_ch4_files = np.array(sorted(all_ch4_files))
        ch4_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_CH4_BadPixelMap_CH4_20250722.nc','r')['BadPixelMap'][:] 
        ch4_pfx_list=open('noisy_ch4_id').readlines()
        ch4_files = []
        for ipfx,pfx in enumerate(ch4_pfx_list):
            ch4_pfx_list[ipfx] = pfx.strip('\n')
            print(f'Dark ID {ipfx}: {ch4_pfx_list[ipfx]}')
            ch4_inds = np.where([str(s).split('/')[3] == f'{ch4_pfx_list[ipfx]}' for s in all_ch4_files])[0].flatten()
            if len(ch4_inds) == 0:
                print('Warning: {pfx} had no matching CH4 files')
                continue
            ch4_files.extend(all_ch4_files[ch4_inds])
            
        all_o2_files = list(Path('./2024/').rglob('*O2*.nc'))
        all_o2_files.extend(list(Path('./2025/').rglob('*O2*.nc')))
        all_o2_files = np.array(sorted(all_o2_files))
        o2_bp = Dataset('../level1a_calibration_MSAT_20250722.0.0_O2_BadPixelMap_O2_20250722.nc','r')['BadPixelMap'][:] 
        o2_pfx_list=open('noisy_o2_id').readlines()
        o2_files = []
        for ipfx,pfx in enumerate(o2_pfx_list):
            o2_pfx_list[ipfx] = pfx.strip('\n')
            print(f'Dark ID {ipfx}: {o2_pfx_list[ipfx]}')
            o2_inds = np.where([str(s).split('/')[3] == f'{o2_pfx_list[ipfx]}' for s in all_o2_files])[0].flatten()
            if len(o2_inds) == 0:
                print('Warning: {pfx} had no matching O2 files')
                continue
            o2_files.extend(all_o2_files[o2_inds])
            
        ch4_args = [{'ID':str(fi),'file':str(fi),'fpa':'CH4'} for fi in ch4_files]
        run_function_in_parallel(plot_all_frames,ch4_args)
        o2_args = [{'ID':str(fi),'file':str(fi),'fpa':'O2'} for fi in o2_files]
        run_function_in_parallel(plot_all_frames,o2_args)
