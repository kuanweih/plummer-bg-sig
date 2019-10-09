import numpy as np
import pandas as pd

from src.plummer_bg_sig import *


# fac_sigma2 = 0.68
fac_sigma2 = 0.5


if __name__ == '__main__':
    print('Loading dwarf data...')
    df = pd.read_csv('dwarfs_detail.csv').drop(columns=['Unnamed: 0'])
    df = df[['GalaxyName', 'Nstar_rh', 'rh_deg', 'Nbg_per_deg2', 'Nstar_per_rhdeg3']]
    # df = df.loc[(df.GalaxyName == 'Antlia2') |
    #             (df.GalaxyName == 'Fornax') |
    #             (df.GalaxyName == 'Sculptor') |
    #             (df.GalaxyName == 'LeoI')]

    n_dwarf = df.shape[0]
    print('There are %d dwarfs in the csv' % int(n_dwarf))

    list_bgres, list_sigs = [], []
    keys_bgres, keys_sigs = [], []

    dict_analysis = {}

    for i in range(n_dwarf):
        print('\n--------------------------------------------------\n')
        print(df['GalaxyName'][i])

        rh = df['rh_deg'][i]
        nstar = df['Nstar_rh'][i]

        sigma1 = rh / 10
        sigma2 = fac_sigma2 * rh
        sigma3 = 0.5

        rmax = max(10 * rh, rh + 3.*sigma3)

        width = 2 * rmax
        pixelsize = sigma1 / 2.
        surface_density = df['Nbg_per_deg2'][i]
        nbg = round((2*rmax)**2 * surface_density)


        res_dict, sigs_dict = main(rh, nstar, rmax, sigma1, sigma2, sigma3,
                                   width, pixelsize, surface_density,
                                   is_plot=False, is_detail=False)

        elements = ['_min', '_max', '_mean', '_std']

        if i==0:
            for j,e in enumerate(elements):
                for k,v in sigs_dict.items():
                    dict_analysis[k + e] = [v[j]]
                for k,v in res_dict.items():
                    dict_analysis[k + e] = [v[j]]
        else:
            for j,e in enumerate(elements):
                for k,v in sigs_dict.items():
                    dict_analysis[k + e].append(v[j])
                for k,v in res_dict.items():
                    dict_analysis[k + e].append(v[j])

    print('\n--------------------------------------------------\n')

    df_analysis = pd.DataFrame.from_dict(dict_analysis)
    df = pd.concat([df, df_analysis], axis=1)
    df.to_csv('sig_test_s2_%s_rh.csv' % str(fac_sigma2))
