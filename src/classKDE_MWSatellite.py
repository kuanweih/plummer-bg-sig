import time
import numpy as np


from typing import Tuple
from src.tools import dist2
from scipy.stats import poisson
from scipy.special import erfcinv, erfc
from scipy.signal import fftconvolve, gaussian



class KDE_MWSatellite(object):
    def __init__(self, ra_center: float, dec_center: float, width: float,
            ps: float, sigma1: float, sigma2: float, sigma3: float, rh: float):
        """ Kernel Density Estimation on a PatchMWSatellite object. This class
        contains 2 kernels: Gaussian and Poisson.

        : ra_center : ra of the center of the patch in deg
        : dec_center : dec of the center of the patch in deg
        : width : width of the patch in deg
        : ps : size of pixel in deg
        : sigma1 : target kernel size in deg: GCs
        : sigma2 : background kernel size inside the satellite in deg
        : sigma3 : background kernel size outside the satellite in deg
        : rh : half-light radius in deg
        """
        self.ra_center = ra_center
        self.dec_center = dec_center
        self.width = width
        self.pixel_size = ps
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.rh = rh

        self.num_grid = round(self.width / self.pixel_size)
        self.x_mesh = self.grid_coord(self.ra_center)
        self.y_mesh = self.grid_coord(self.dec_center)

    def __str__(self):
        s1 = "This is a KDE_MWSatellite object: \n"
        s2 = "    pixel size = %0.8f\n" % self.pixel_size
        s3 = "    number of grids = %d\n" % self.num_grid
        s4 = "    sigma1 = %0.8f deg\n" % self.sigma1
        s5 = "    sigma2 inside the satellite = %0.8f deg\n" % self.sigma2
        s6 = "    sigma2 outside the satellite = %0.8f deg\n" % self.sigma3
        s7 = "    rh = %0.8f deg" % self.rh
        return "{}{}{}{}{}{}{}".format(s1, s2, s3, s4, s5, s6, s7)

    def grid_coord(self, center: float) -> np.ndarray:
        """ Get grid coordinates according to the center position and the
        width of the mesh.
        : center : the center of the coordinates
        : return : mesh coordinates array
        """
        _min = center - 0.5 * self.width
        _max = center + 0.5 * self.width
        return np.linspace(_min, _max, num=self.num_grid, endpoint=True)

    def np_hist2d(self, ra: np.ndarray, dec: np.ndarray):
        """ Get histogram 2d for the star distribution on the mesh
        : ra : PatchMWSatellite.datas['ra']
        : dec : PatchMWSatellite.datas['dec']
        """
        self.hist2d, _, _ = np.histogram2d(
            dec, ra, bins=(self.y_mesh, self.x_mesh))
        print('Added hist2d according to the sources on the patch.')

    def add_masks_on_pixels(self, ra_df: float, dec_df: float, radius: float):
        """ Get histogram 2d for the star distribution on the mesh

        : ra_df : ra of the dwarf
        : dec_df : dec of the dwarf
        : radius : the radius telling inside or outside
        """
        x2d, y2d = np.meshgrid(self.x_mesh, self.y_mesh)
        # reshape the array to be like hist2d
        x2d = 0.5 * (x2d[1:, 1:] + x2d[:-1, :-1])
        y2d = 0.5 * (y2d[1:, 1:] + y2d[:-1, :-1])
        _dist2 = dist2(x2d, y2d, ra_df, dec_df)
        self.is_inside_dwarf = _dist2 < radius**2
        print('Added a mask array telling if pixels are inside the dwarf. \n')
        self.is_overlap = _dist2 < (radius + self.sigma3)**2
        self.is_overlap = self.is_overlap & (~self.is_inside_dwarf)
        print('Added a mask array telling if pixels overlap the dwarf and the outer aperture. \n')

    def fftconvolve_boundary_adjust(
            self, maps_ori: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ Use scipy signal fftconvolve to calculate the convolved map.
        Edge effect is also taken care of. Using fftconvolve will yeild
        some negative elements while they are actuaclly 0 when using
        convolve. Also, there will be some positive noises which need to
        be taken care of. Therefore, conv[conv < 1e-15] = 0 is applied
        to get rid of the false divisions.

        : maps_ori : original maps matrix
        : kernel : kernel matrix
        : return : convolved maps with non negative values
        """
        conv = fftconvolve(maps_ori, kernel, mode='same')
        mask2d = np.ones(maps_ori.shape)
        conv /= fftconvolve(mask2d, kernel, mode='same')
        conv[conv < 1e-15] = 0.    # rounding the noise < 1e-15
        return conv

    def overdensity(self, sigma: float) -> np.ndarray:
        """ Convolved overdensity maps with a Gaussian kernel size sigma """
        s_grid = sigma / self.pixel_size
        truncate = 5    # TODO: parameterize it later
        kernel_grid = int(truncate * s_grid)
        _gaussian = gaussian(kernel_grid, s_grid)
        kernel = np.outer(_gaussian, _gaussian) / 2. * np.pi * s_grid**2
        return self.fftconvolve_boundary_adjust(self.hist2d, kernel)

    def get_sig_gaussian(self, od_1: np.ndarray, od_2: np.ndarray,
                         sigma1: float, sigma2: float) -> np.ndarray:
        """ Get a significance maps using the 2-gaussian kernel
        : od_1 : overdensity with sigma1
        : od_2 : overdensity with sigma2
        : sigma1 : inner kernel
        : sigma2 : outer kernel
        : return : significance from 2-gaussian kernel density estimation
        """
        s1 = sigma1 / self.pixel_size
        sig = (od_1 - od_2) * np.sqrt(4. * np.pi * s1**2)
        return np.divide(sig, np.sqrt(od_2),
                         out=np.zeros_like(sig), where=od_2 != 0)  # force 0/0 = 0

    def compound_sig_gaussian(self):
        """ Compound the Gaussian significance map: s12 inside (s23 > sigma_th)
        and s13 outside (s23 < sigma_th) """
        t0 = time.time()

        od_1 = self.overdensity(self.sigma1)
        od_2 = self.overdensity(self.sigma2)
        od_3 = self.overdensity(self.sigma3)
        s12 = self.get_sig_gaussian(od_1, od_2, self.sigma1, self.sigma2)
        s13 = self.get_sig_gaussian(od_1, od_3, self.sigma1, self.sigma3)

        self.sig_gaussian = s12 * self.is_inside_dwarf + s13 * (~self.is_inside_dwarf)
        print("Took %0.4fs to calculate Gaussian sig." % (time.time() - t0))
        print('Added sig_gaussian to the KDE_MWSatellite object. \n')

    def z_score_poisson(self, lamb: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ Calculate the z-score of the tail probability of poisson via N(0, 1)
        according to z = sqrt(2) * erfinv(1 - 2 * sf(x, lambda)), where sf
        (survival function) = 1 - CDF.
        : lamb : expected background number count from outer aperture (lambda)
        : x : number count of observed stars in the inner aperture
        : return : z-score of poisson map
        """
        return np.sqrt(2.) * erfcinv(2. * poisson.sf(x, lamb))

    def circular_kernel(self, radius: int) -> np.ndarray:
        """ Calculate the circular kernel with radius according to sigma. This
        kernel is not normalized because we are using it to sum over the number
        count """
        radius = int(radius)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x**2 + y**2 <= radius**2
        kernel[mask] = 1
        return kernel

    def poisson_inner_number_count(self,
                                   sigma: float) -> Tuple[np.ndarray, float]:
        """ Calculate inner number count of stars within the area of radius
        of sigma, using convolution.
        : return : convolved maps (number count of the inner aperture)
        : return : number of pixels of the inner aperture
        """
        s_grid = sigma / self.pixel_size
        kernel = self.circular_kernel(round(s_grid))
        norm_kernel = np.sum(kernel)
        conv = self.fftconvolve_boundary_adjust(self.hist2d,
                                                (kernel / norm_kernel))
        return conv * norm_kernel, norm_kernel

    def poisson_outer_expected_background(self, sigma_in: float,
            sigma_out: float, is_overlap: bool = False) -> Tuple[np.ndarray, float]:
        """ Calculate expected backgound number count of stars based on the
        outer aperture, which will be using to calculate 'lambda' for poisson.
        : sigma_in : inner radius of the outer aperture
        : sigma_out : outer radius of the outer aperture
        : is_overlap : if outer aperture is overlapping with the dwarf
        : return : convolved maps (backgound estimation)
        : return : number of pixels of the outer aperture
        """
        s_grid_in = int(round(sigma_in / self.pixel_size))
        s_grid_out = int(round(sigma_out / self.pixel_size))
        kernel_out = self.circular_kernel(s_grid_out)
        ds_pad = s_grid_out - s_grid_in
        kernel_in_pad = np.pad(self.circular_kernel(s_grid_in),
                               ds_pad, 'constant', constant_values=0)
        kernel = kernel_out - kernel_in_pad
        norm_kernel = np.sum(kernel)

        if is_overlap:
            conv = self.fftconvolve_boundary_adjust(
                (self.hist2d * (~self.is_inside_dwarf)), (kernel / norm_kernel))
            area_overlap = self.fftconvolve_boundary_adjust(
                self.is_inside_dwarf, (kernel / norm_kernel))
            area_overlap = norm_kernel * area_overlap
            return conv * norm_kernel, norm_kernel - area_overlap

        conv = self.fftconvolve_boundary_adjust(
            self.hist2d, (kernel / norm_kernel))
        return conv * norm_kernel, norm_kernel

    def get_lambda_poisson(self, n_o: np.ndarray, area_o: np.ndarray,
                           area_i: np.ndarray) -> np.ndarray:
        """ Calculate lambda as the estimated background number count.
        : n_o : number of sources from outer aperture
        : area_o : area of outer aperture
        : area_i : area of inner aperture
        : return : lambda (estimated background count) = n_o * area_i / area_o
        """
        return n_o * area_i / area_o

    def compound_sig_poisson(self):
        """ Compound the Poisson significance map: s12 inside (s23 > sigma_th)
        and s13 outside (s23 < sigma_th)
        """
        t0 = time.time()

        # factors using for outer aperture
        f_in2out = 2.    # r_i = f_in2out * s1
        rh_th = 10. * self.sigma1    # threshold of min half-light radius in deg

        # inner aperture
        n_inner, area_inner = self.poisson_inner_number_count(self.sigma1)

        # outer aperture outside of the dwarf
        r_i = f_in2out * self.sigma1
        r_o = self.sigma3
        n_outer, area_outer = self.poisson_outer_expected_background(r_i, r_o)

        if np.sum(self.is_overlap) > 0:    # if there are overlapping pixels
            n_outer_no_dwarf, area_outer_no_dwarf = self.poisson_outer_expected_background(
                r_i, r_o, is_overlap=True)
            n_outer = n_outer_no_dwarf * self.is_overlap + n_outer * (~self.is_overlap)
            area_outer = area_outer_no_dwarf * self.is_overlap + area_outer * (~self.is_overlap)

        lambda_out = self.get_lambda_poisson(n_outer, area_outer, area_inner)

        # outer aperture inside the dwarf
        if self.rh < rh_th:
            lambda_in = lambda_out    # TODO check this later
        else:
            r_o = self.sigma2
            n_outer, area_outer = self.poisson_outer_expected_background(
                r_i, r_o)
            lambda_in = self.get_lambda_poisson(
                n_outer, area_outer, area_inner)

        s12 = self.z_score_poisson(lambda_in, n_inner)
        s13 = self.z_score_poisson(lambda_out, n_inner)

        self.sig_poisson = s12 * self.is_inside_dwarf + \
            s13 * (~self.is_inside_dwarf)
        print("Took %0.4fs to calculate Poisson sig." % (time.time() - t0))
        print('Added sig_poisson to the KDE_MWSatellite object. \n')

        self.bg_estimate = lambda_in * self.is_inside_dwarf + \
            lambda_out * (~self.is_inside_dwarf)
        self.bg_estimate /= area_inner

    def true_bg_parser(self, true_bg: np.ndarray):
        """ Parse true_bg in to be used as self.hist2d
        : true_bg : the true or analytic bg
        """
        self.hist2d = true_bg

    def true_const_sig_parser(self, true_const_sig: float):
        """ Parse true_const_sig in to generate a 2d maps
        : true_const_sig : the true constant significance value
        """
        self.true_sig = true_const_sig * np.ones(self.hist2d.shape)

    def inv_z_score_poisson(self, lamb: np.ndarray, z: np.ndarray) -> np.ndarray:
        """ Calculate the inverse of z-score of
        z = np.sqrt(2.) * erfcinv(2. * poisson.sf(x, lamb))

        : lamb : expected background number count from outer aperture (lambda)
        : z : z-score of poisson map
        : return : number count of observed stars in the inner aperture
        """
        return  poisson.isf(0.5 * erfc(z / np.sqrt(2.)), lamb)
