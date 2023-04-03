import numpy as np

import astropy
import astropy.units as u
import astropy.coordinates
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from skimage.transform import warp_polar
from scipy.ndimage import map_coordinates


def pearson_mat(matrix, target):
    """
    Assume matrix is of size (samples, x, y)
    and target is of size (samples, 1)
    """

    if len(target.shape) < 3:
        target.shape += (1,) * (3-len(target.shape))
    avg_matrix = matrix.mean(axis=0)
    avg_target = target.mean(axis=0)
    avg_diff_mat = matrix - avg_matrix
    avg_diff_tar = target - avg_target
    term1 = np.sum(avg_diff_mat * avg_diff_tar, axis=0)
    term2 = np.sqrt(np.sum(avg_diff_mat**2, axis=0))
    term3 = np.sqrt(np.sum(avg_diff_tar**2, axis=0))
    term4 = np.fmax(term2*term3, 1e-4*np.ones(term2.shape))
    return term1/term4


def spearman_mat(matrix, target):
    rank_x = np.argsort(matrix, axis=0)
    rank_y = np.argsort(target, axis=0)
    return pearson_mat(rank_x, rank_y)


def range_tuple(matrix):
    return np.nanmin(matrix), np.nanmax(matrix)

def chunks(l, chunk_size):
    for i in range(0, len(l), chunk_size):
        yield l[i:i+chunk_size]
    
    return None

def map_to_polar(sun_map, out_shape = (360,100)):

    # get origin and leftmost point of the solar disk in arcseconds
    origin = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=sun_map.coordinate_frame)
    left_rad_pos = SkyCoord(sun_map.rsun_obs, 0*u.arcsec, frame=sun_map.coordinate_frame)

    # transform to pixel position and get only x component
    origin_pixel, _ = skycoord_to_pixel(origin, sun_map.wcs)
    left_rad_pixel, _ = skycoord_to_pixel(left_rad_pos, sun_map.wcs)

    # calculate the radius in pixels
    pixel_rad_len = left_rad_pixel - origin_pixel

    # transform the data in the map to polar
    polar_map = warp_polar(sun_map.data, radius=np.ceil(pixel_rad_len), output_shape=out_shape)
    polar_map = np.rot90(polar_map, -1) # put the center on the bottom

    return polar_map


def polar_linear(img, o=None, r=None, output=None, order=1, cont=0, cval=0):
    """
    Taken from https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2
    """

    

    if img.ndim == 3:
        if r is None: 
            r = img.shape[0]

        output_original = output

        if output is None:
            output = np.zeros((r*2, r*2, img.shape[2]), dtype=img.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=img.dtype)

        for i in range(img.shape[2]):
            output[:,:,i] = polar_linear(img[:,:,i], o, r, output_original, order, cont, cval)
            
    elif img.ndim == 2:
        if r is None: 
            r = img.shape[0]

        if output is None:
            output = np.zeros((r*2, r*2), dtype=img.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=img.dtype)

        if o is None: 
            o = np.array(output.shape)/2 - 0.5

        out_h, out_w = output.shape
        ys, xs = np.mgrid[:out_h, :out_w] - o[:,None,None]
        rs = (ys**2+xs**2)**0.5
        ts = np.arccos(xs/rs)
        ts[ys<0] = np.pi*2 - ts[ys<0]
        ts *= (img.shape[1]-1)/(np.pi*2)

        map_coordinates(img, (rs, ts), order=order, output=output, cval=cval)

        output = np.flip(output, axis=0)

    return output


    
