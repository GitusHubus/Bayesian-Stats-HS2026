import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass

#Constants
DATA = Path(__file__).parent / "Data"    #Data
DATA_ELLS = ("0", "2", "4",)              #Multipole ells present in DATA

#Define Class that handels data outputs
@dataclass
class DESI_DATA:
    #Settings
    tag : str
    ells : tuple

    #Data
    k_vec : np.ndarray
    ell_vec : np.ndarray
    k_theory : dict
    data : np.ndarray
    C : np.ndarray
    Cinv : np.ndarray
    W : np.ndarray

    @classmethod
    def load_desi(cls, tag, ells):
        # 0. Check input
        assert all(ell in DATA_ELLS for ell in ells), f"ells must be a subset of {DATA_ELLS}, as strings; got {ells}"
        # 1. File paths
        spectrum_file, cov_file, win_file = load_files(tag)

        # 2. Read everything (all multipoles in the files)
        k, k_edges, data_full       = get_spectrum(spectrum_file)
        cov_edges, C_full           = get_cov(cov_file)
        k_theory, win_edges, W_full = get_win(win_file)

        # 3. Cut to the multipoles in `ells`
        data, C, Cinv, W, data_mask = mask_data(ells, k_edges, data_full,
                                            cov_edges, C_full, win_edges, W_full)

        # 4. Labels for each data entry
        k_vec   = np.concatenate([k[ell] for ell in DATA_ELLS])[data_mask]
        ell_vec = np.concatenate([np.full(len(k[ell]), ell) for ell in DATA_ELLS])[data_mask]

        # 5. Essential checks
        n = len(data)
        assert C.shape == (n, n) and W.shape[0] == n
        assert np.allclose(C, C.T)
        assert np.linalg.eigvalsh(C).min() > 0

        # 6. Package
        return DESI_DATA(tag=tag, ells=ells, k_vec=k_vec, ell_vec=ell_vec,
                                 k_theory=k_theory, data=data, C=C, Cinv=Cinv, W=W) #DESI_DATA contains all relevant Data

#Load Desi files from folder
def load_files(tag):

    spectrum_file = DATA / ("spectrum-poles-rotated-corrected_" + tag + ".h5")
    cov_file = DATA / ("covariance_spectrum-poles-rotated_" + tag + ".h5")
    win_file = DATA / ("window_spectrum-poles-rotated_" + tag + ".h5")

    return spectrum_file, cov_file, win_file

#Extract spectrum Data
def get_spectrum(file):
    
    k, k_edges, P = {}, {}, {}

    with h5py.File(file) as h:               # step 1: open the file on disk
        for ell in DATA_ELLS:                              # step 2: Run over all ells
            k[ell] = h[f"{ell}/k"][:]                 # step 3: read the dataset "ell/k", ell = 0,2,4
            k_edges[ell] = h[f"{ell}/k_edges"][:]     # step 4: read the dataset "ell/k_edges", ell = 0,2,4
            P[ell] = h[f"{ell}/value"][:]             # step 5: read the dataset "ell/value", ell = 0,2,4

    data_full = np.concatenate([P[ell] for ell in DATA_ELLS])   # Data should be a vector to calculate r = data - model => chi^2
                                                                # First 36 entries are ell = 0, then ell = 2, then ell = 4
    return k, k_edges, data_full
 
#Extract covariance Data
def get_cov(file):

    cov_edges = {}


    with h5py.File(file) as h:
        for ell in DATA_ELLS:
            cov_edges[ell] = h[f"observable/{ell}/k_edges"][:] #Get k_edges for the covariance matrix
    
        C_full = h["value"][:]                                 # Get Covariance Matrix

    return cov_edges, C_full

#Extract window Data
def get_win(file):
    
    win_edges, k_theory = {}, {}


    with h5py.File(file) as h:
        for ell in DATA_ELLS:
            win_edges[ell] = h[f"observable/{ell}/k_edges"][:] 
            k_theory[ell] = h[f"theory/{ell}/k"][:]
        
        W_full = h["value"][:]                                

    return k_theory, win_edges, W_full

#Mask: Cuts data to correct size
def make_mask(ells, data_edges, file_edges):                      #file_edges = k_edges, cov_edges, win_edges
    
    mask = []
    
    for ell in DATA_ELLS:
        for e in file_edges[ell]:
            if ell in ells:
                match = np.isclose(e,data_edges[ell]).all(axis=1).any()
            else:
                match = False
            mask.append(match)
            
    return np.array(mask)

#Final Data: Only part of Data we need
def mask_data(ells, k_edges, data_full, cov_edges, C_full, win_edges, W_full):
    
    data_mask = make_mask(ells, k_edges, k_edges)                              
    cov_mask = make_mask(ells, k_edges, cov_edges)
    win_mask = make_mask(ells, k_edges, win_edges)
    
    data      = data_full[data_mask]
    C = C_full[np.ix_(cov_mask,cov_mask)]
    C_inv = np.linalg.inv(C) 
    W = W_full[win_mask,:]

    return data, C, C_inv, W, data_mask