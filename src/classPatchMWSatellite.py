import numpy as np
import sqlutilpy

from src.tools import dist2



class PatchMWSatellite(object):
    def __init__(self, name_sat: str, ra_sat: float, dec_sat: float,
                 dist: float, width: float, database: str, catalog_str: str):
        """ Milky Way (MW) Satellite object within a patch.

        : name_sat : name of the satellite, e.g. Fornax
        : ra_sat : RA of the satellite in deg
        : dec_sat : Dec of the satellite in deg
        : dist : distance of the satellite (pc)
        : width : width of the square area when querying data in deg
        : database : database to be queried
        : catalog_str : a string of catalogs for querying
        """
        self.name_sat = name_sat
        self.ra_sat = ra_sat
        self.dec_sat = dec_sat
        self.dist = dist
        self.width = width
        self.database = database
        self.catalog_str = catalog_str
        self.catalog_list = catalog_str.replace("\n", "").replace(" ", "").split(",")
        self.datas = {}

    def __str__(self):
        str1 = "This is a PatchMWSatellite object:\n"
        str2 = "    name = {}\n".format(self.name_sat)
        str3 = "    ra = {}\n    dec = {}\n".format(self.ra_sat, self.dec_sat)
        str4 = "    map width = {} deg\n".format(self.width)
        str5 = "    database = {}\n".format(self.database)
        return  "{}{}{}{}{}".format(str1, str2, str3, str4, str5)

    def n_source(self) -> int:
        """ Calculate the number of stars in the patch """
        return  len(self.datas[self.catalog_list[0]])

    def sql_get(self, host: str, user: str, password: str):
        """ Query 'catalog_str' from 'database' using sqlutilpy.get() """
        ra_min = self.ra_sat - 0.5 * self.width
        ra_max = self.ra_sat + 0.5 * self.width
        dec_min = self.dec_sat - 0.5 * self.width
        dec_max = self.dec_sat + 0.5 * self.width

        query_str = """
                    select {} from {}
                    where {} < ra and ra < {} and {} < dec and dec < {}
                    """.format(self.catalog_str, self.database,
                               ra_min, ra_max, dec_min, dec_max)

        print("Querying data in the patch using sqlutilpy.get():")
        datas = sqlutilpy.get(query_str,
                              host=host, user=user, password=password)

        # update 'datas' dic to store queried data
        for i, catalog in enumerate(self.catalog_list):
            self.datas[catalog] = datas[i]
        print("    %d sources are queried \n"  %self.n_source())

    def cut_datas(self, mask: np.ndarray):
        """ Cut datas based on the mask. """
        for key, column in self.datas.items():
            self.datas[key] = column[mask]

    def mask_cut(self, catalog: str, min_val: float, max_val: float):
        """ Cut the data with a min and a max value """
        print("Applying a cut: {} < {} < {}:".format(min_val, catalog, max_val))
        maskleft = min_val < self.datas[catalog]
        maskright = self.datas[catalog] < max_val
        mask = maskleft & maskright
        self.cut_datas(mask)
        print("    %d sources left \n"  %self.n_source())

    def mask_g_mag_astro_noise_cut(self):
        """ Hard code the astrometric_excess_noise and phot_g_mean_mag cut """
        print("Applying astrometric_excess_noise and phot_g_mean_mag cut.")
        noise = self.datas["astrometric_excess_noise"]
        g_mag = self.datas["phot_g_mean_mag"]
        maskleft = (g_mag <= 18.) & (noise < np.exp(1.5))
        maskright = (18. < g_mag) & (noise < np.exp(1.5 + 0.3 * (g_mag - 18.)))
        mask = maskleft | maskright
        self.cut_datas(mask)
        print("    %d sources left \n"  %self.n_source())

    def mask_panstarrs_stargalaxy_sep(self):
        """ Hard code the star galaxy separation """
        print("Applying star galaxy separation: (rpsfmag - rkronmag) < 0.05")
        rpsfmag = self.datas["rpsfmag"]
        rkronmag = self.datas["rkronmag"]
        mask = (rpsfmag - rkronmag) < 0.05
        self.cut_datas(mask)
        print("    %d sources left \n"  %self.n_source())

    def append_is_inside(self, ra_df: float, dec_df: float, radius: float):
        """ Assign a boolean value to specify if a source is in the area
        within the radius from the dwarf.

        : ra_df : ra of the dwarf
        : dec_df : dec of the dwarf
        : radius : the radius telling inside or outside
        """
        _dist2 = dist2(self.datas['ra'], self.datas['dec'], ra_df, dec_df)
        self.datas['is_inside'] = np.array(_dist2 < radius ** 2)
        print('Appended a boolean array telling is_inside. \n')

    def append_sig_to_data(self, x_mesh, y_mesh, sig_gaussian, sig_poisson):
        """ Append significance of each star to the datas """
        pixel_size_x = np.max(np.diff(x_mesh))
        pixel_size_y = np.max(np.diff(y_mesh))
        id_xs = (self.datas["ra"] - x_mesh[0]) / pixel_size_x
        id_ys = (self.datas["dec"] - y_mesh[0]) / pixel_size_y

        # is_insides = []
        sig_gaussian_stars = []
        sig_poisson_stars = []
        for i in range(self.n_source()):
            id_x, id_y = int(id_xs[i]), int(id_ys[i])
            # is_insides.append(self.is_inside[id_y][id_x])
            sig_gaussian_stars.append(sig_gaussian[id_y][id_x])
            sig_poisson_stars.append(sig_poisson[id_y][id_x])

        # self.datas["is_inside"] = np.array(is_insides)
        self.datas["sig_gaussian"] = np.array(sig_gaussian_stars)
        self.datas["sig_poisson"] = np.array(sig_poisson_stars)
