import h5py
import numpy as np

with h5py.File('linear_better/linear_better.h5', 'r') as f:
    for ds_name in ['k_0_circ_1', 'k_1_circ_100', 'k_2_circ_200', 'k_3_circ_300']:
        if ds_name not in f:
            continue
        print(f'\n=== {ds_name} ===')
        t = np.array(f[f'{ds_name}/times'])
        a = np.array(f[f'{ds_name}/a'])
        e = np.array(f[f'{ds_name}/e'])
        
        # Look at first 10 values and check for anomalies
        print(f'a[:10] = {a[:10]}')
        print(f'e[:10] = {e[:10]}')
        
        # Check the relative difference between consecutive points
        a_diff = np.abs(np.diff(a[:10]))
        a_rel_diff = a_diff / a[:9]
        print(f'Relative a diff (first 9): {a_rel_diff}')
        
        # Same for e
        e_diff = np.abs(np.diff(e[:10]))
        e_rel_diff = e_diff / e[:9]
        print(f'Relative e diff (first 9): {e_rel_diff}')
