import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import special
from src.tools import dist2
from src.classKDE_MWSatellite import KDE_MWSatellite


def plummer2d(a, x, y, fac_plummer):
    return fac_plummer * (4./3.) * a**5 / (a**2 + x**2 + y**2)**2


def detail_single_dwarf(dwarfname: str, df_all: pd.DataFrame):
    df_dwarf = df_all.loc[df_all['GalaxyName'] == dwarfname]

    rh = float(df_dwarf['rh_deg'])    # half light radius of the plummer
    nstar = int(df_dwarf['Nstar_rh'])    # number of stars of the plummer
    rmax = 10 * rh
    # fac_plummer = 3. * nstar / 4. / np.pi / rh**3

    sigma1 = 0.01
    sigma2 = 0.68 * rh
    sigma3 = 0.5

    width = 2 * rmax
    pixelsize = sigma1

    surface_density = float(df_dwarf['Nbg_per_deg2'])    # # of stars per deg^2
    nbg = round((2*rmax)**2 * surface_density)

    print(dwarfname)
    print('\nhalf-light radius: %0.1f deg' % rh)
    print('number of stars of the Plummer: %d' % nstar)
    print('kernel size of outer aperture outside of the Plummer: %0.2f deg' % sigma3)
    print('surface density of uniform background = %d per deg^2 \n' % surface_density)

    main(rh, nstar, rmax, sigma1, sigma2, sigma3, width, pixelsize, surface_density)



def main(rh, nstar, rmax, sigma1, sigma2, sigma3,
         width, pixelsize, surface_density, is_plot=True, is_detail=True):

    ra_center_patch = 0.
    dec_center_patch = 0.

    ra_dwarf = 0.
    dec_dwarf = 0.

    aplummer = rh / 1.3
    fac_plummer = 3. * nstar / 4. / np.pi / rh**3

    xedges = np.linspace(-rmax, rmax)
    yedges = np.linspace(-rmax, rmax)
    extent = [xedges.min(), xedges.max(), yedges.min(), yedges.max()]


    kdepatch = KDE_MWSatellite(ra_center_patch, dec_center_patch,
                               width, pixelsize, sigma1, sigma2, sigma3, rh)

    xmesh, ymesh = np.meshgrid(kdepatch.x_mesh, kdepatch.y_mesh)
    xmesh = 0.5 * (xmesh[1:, 1:] + xmesh[:-1, :-1])
    ymesh = 0.5 * (ymesh[1:, 1:] + ymesh[:-1, :-1])
    dist2_ = dist2(xmesh, ymesh, ra_dwarf, dec_dwarf)

    xplummer = xmesh - ra_dwarf
    yplummer = ymesh - dec_dwarf
    true_bg = plummer2d(aplummer, xplummer, yplummer, fac_plummer) * pixelsize**2
    true_bg += surface_density * pixelsize**2
    kdepatch.true_bg_parser(true_bg)
    kdepatch.add_masks_on_pixels(ra_dwarf, dec_dwarf, rh)

    kdepatch.compound_sig_poisson()
    lambda_out = kdepatch.bg_estimate


    mask_out_rh = dist2_ > rh**2
    mask_in_boundary = dist2_ < (rmax - sigma3)**2
    mask_in_sigma3 = dist2_ < (rh + sigma3)**2


    true_sigs = [2, 5, 8]
    sigs_dict = {}
    for true_sig in true_sigs:
        x = kdepatch.inv_z_score_poisson(true_bg, true_sig)
        sig_estimate = kdepatch.z_score_poisson(kdepatch.bg_estimate, x)
        sig_estimate *= mask_in_boundary

        if is_plot:
            plot_sig(true_sig, sig_estimate, mask_in_boundary, extent, rh, sigma3)

        _s = sig_estimate[~mask_out_rh].flatten()
        sigs_dict['in_dwarf_s%d' % true_sig] = np.array([
            np.min(_s), np.max(_s), np.mean(_s), np.std(_s)])
        _s = sig_estimate[mask_out_rh & mask_in_sigma3 & mask_in_boundary].flatten()
        sigs_dict['overlap_s%d' % true_sig] = np.array([
            np.min(_s), np.max(_s), np.mean(_s), np.std(_s)])
        _s = sig_estimate[~mask_in_sigma3 & mask_in_boundary].flatten()
        sigs_dict['outskirt_s%d' % true_sig] = np.array([
            np.min(_s), np.max(_s), np.mean(_s), np.std(_s)])
        del _s


    res_out = lambda_out - true_bg
    res_out /= true_bg
    res_out *= mask_in_boundary

    if is_plot:
        lambda_out *= mask_in_boundary
        plot_detail(true_bg, lambda_out, res_out, mask_out_rh,
                    mask_in_boundary, mask_in_sigma3, extent, rh, sigma1, sigma3)

    del lambda_out


    res_mean = np.mean(res_out)
    res_std = np.std(res_out)

    res_dict = {}
    _s = res_out[~mask_out_rh].flatten()
    res_dict['in_dwarf_res_nbg'] = np.array([
        np.min(_s), np.max(_s), np.mean(_s), np.std(_s)])
    _s = res_out[mask_out_rh & mask_in_sigma3 & mask_in_boundary].flatten()
    res_dict['overlap_res_nbg'] = np.array([
        np.min(_s), np.max(_s), np.mean(_s), np.std(_s)])
    _s = res_out[~mask_in_sigma3 & mask_in_boundary].flatten()
    res_dict['outskirt_res_nbg'] = np.array([
        np.min(_s), np.max(_s), np.mean(_s), np.std(_s)])


    if not is_plot:
        return res_dict, sigs_dict


def plot_detail(true_bg, lambda_out, res_out, mask_out_rh,
                mask_in_boundary, mask_in_sigma3, extent, rh, sigma1, sigma3):

    sns.set(style="white", color_codes=True, font_scale=1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))


#     im = ax[0].imshow(true_bg * mask_out_rh,
    im = ax[0].imshow(true_bg,
                      cmap='Blues', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('true bg')


#     im = ax[1].imshow(lambda_out * mask_out_rh,
    im = ax[1].imshow(lambda_out,
                      cmap='Blues', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title('bg estimation')

    _vmax, _vmin = res_out.flatten().max(), res_out.flatten().min()
    _vabs = max(_vmax, -_vmin)
    im = ax[2].imshow(res_out, vmin=-_vabs, vmax=_vabs,
                      cmap='coolwarm', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax[2])
    ax[2].set_title('(bg_est - bg_true) / bg_true')

    circle = plt.Circle((0, 0), rh, color='k', fill=False, lw=2, alpha=0.6)
    ax[2].add_artist(circle)

    circle = plt.Circle((0, 0), rh+sigma3, color='k', fill=False, lw=2, alpha=0.6)
    ax[2].add_artist(circle)



    sns.set(style="white", color_codes=True, font_scale=1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))

    sns.distplot(res_out.flatten(), kde=False, ax=ax[0])
    ax[0].set_yscale('log')

    _xlen, _ylen = res_out.shape
    _xmid = round(0.5 * _xlen)
    _ymid = round(0.5 * _ylen)
    _x = np.linspace(0, extent[1], _xmid)
    _y = res_out[_ymid, -_xmid:]

    sns.lineplot(x=_x, y=_y, ax=ax[1])
    ax[1].axvline(x=0, c='k')
    ax[1].axvline(x=rh, c='k')
    ax[1].axvline(x=1.7/1.3*rh, c='orange')
    ax[1].axvline(x=rh+sigma3, c='k')
    ax[1].axvline(x=extent[1], c='k')



def plot_sig(true_sig, sig_estimate, mask_in_boundary, extent, rh, sigma3):

    sns.set(style="white", color_codes=True, font_scale=1)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    _sig = sig_estimate[mask_in_boundary].flatten()
    _vmax, _vmin = _sig.max(), _sig.min()
    im = ax[0].imshow(sig_estimate, vmin=_vmin, vmax=_vmax,
                      cmap='Blues', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('sig_est of sig_true=%d' % true_sig)
    del _sig

    res = (sig_estimate - true_sig) / true_sig
    res = res * mask_in_boundary
    _vmax, _vmin = res.flatten().max(), res.flatten().min()
    _vabs = max(_vmax, -_vmin)
    im = ax[1].imshow(res, vmin=-_vabs, vmax=_vabs,
                      cmap='coolwarm', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title('(sig_est - sig_true) / sig_true')

    circle = plt.Circle((0, 0), rh, color='k', fill=False, lw=2, alpha=0.6)
    ax[1].add_artist(circle)

    circle = plt.Circle((0, 0), rh+sigma3, color='k', fill=False, lw=2, alpha=0.6)
    ax[1].add_artist(circle)
