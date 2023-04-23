import numpy as np
import sympy

import astropy
import astropy.units as u
import astropy.coordinates
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from skimage.transform import warp_polar
from scipy.ndimage import map_coordinates

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


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

def sunpy_map_to_polar(sun_map, out_shape = (360,100)):

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

def cartesian_to_polar(img, radius=None, out_shape = (360,100)):
    if radius is None:
        radius = img.shape[0]//2
    polar_img = warp_polar(img, radius=radius, output_shape=out_shape)
    polar_img = np.rot90(polar_img, -1) # put the center on the bottom

    return polar_img


def polar_to_cartesian_channels(img, o=None, r=None, output=None, order=1, cont=0, cval=0):
    if r is None: 
        r = img.shape[0]

    output_original = output

    if output is None:
        output = np.zeros((r*2, r*2, img.shape[2]), dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)

    for i in range(img.shape[2]):
        output[:,:,i] = polar_to_cartesian(img[:,:,i], o, r, output_original, order, cont, cval)
    
    return output


def polar_to_cartesian(img, o=None, r=None, output=None, order=1, cont=0, cval=0):
    """
    Taken from https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2
    """
    
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


def linear_to_polar_tf(img_input, radius):
    h, w = radius, radius
    cx, cy = img_input.shape[0]//2, img_input.shape[1]//2

    x, y = tf.meshgrid(tf.linspace(0.0, radius, w), tf.linspace(0.0, radius, h))

    x_trans = x - cx
    y_trans = y - cy
    
    r = tf.sqrt(x_trans**2 + y_trans**2)
    theta = tf.atan2(y_trans, x_trans)

    i = tf.round(r * tf.cos(theta) - cx/2)
    # i = np.clip(i, 0, img_input.shape[0]-1)

    j = tf.round(r * tf.sin(theta) - cy/2)
    # j = np.clip(j, 0, img_input.shape[1]-1)

    print(i, j)
    # out_image = n
    out_image = img_input[tf.cast(i, tf.int32), tf.cast(i, tf.int32)]
    return out_image


def sqauare_dims(size, ratio_w_h=1):
    divs = np.array(sympy.divisors(size))
    dist_to_root = np.abs(divs-np.sqrt(size)*ratio_w_h)
    i = np.argmin(dist_to_root)
    x_size = int(divs[i])
    y_size = size//x_size
    return (x_size, y_size) if x_size < y_size else (y_size, x_size)


def square_dims_vector(vector, ratio_w_h=1):
    return np.reshape(vector.copy(), sqauare_dims(vector.size, ratio_w_h))



# @numba.njit
def abs_max_filter(img, kernel_size=3, cval = 0):
    pad_extra = kernel_size//2
    img_padded = np.full((img.shape[0] + 2*pad_extra, img.shape[1] + 2*pad_extra), cval, dtype=np.float64)
    img_padded[pad_extra:img.shape[0]+pad_extra, pad_extra:img.shape[1]+pad_extra] = img

    kernel = np.empty(kernel_size*kernel_size, dtype=np.float64)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            k_row_start = row
            k_row_end = row + kernel_size

            k_col_start = col
            k_col_end = col + kernel_size

            kernel = img_padded[k_row_start:k_row_end, k_col_start:k_col_end].flatten()
            
            abs_max_idx = np.argmax(np.abs(kernel))
            img[row, col] = kernel[abs_max_idx]
    
    return img

# @numba.jit
def abs_max_filter_par(img, kernel_size=3, cval = 0):
    pad_extra = kernel_size//2
    img_padded = np.full((img.shape[0], img.shape[1] + 2*pad_extra, img.shape[2] + 2*pad_extra), cval, dtype=np.float64)
    img_padded[:, pad_extra:img.shape[1] + pad_extra, pad_extra:img.shape[2] + pad_extra] = img

    kernel = np.empty((img.shape[0], kernel_size**2))
    for row in range(img.shape[1]):
        for col in range(img.shape[2]):
            k_row_start = row
            k_row_end = row + kernel_size

            k_col_start = col
            k_col_end = col + kernel_size

            kernel = img_padded[:, k_row_start:k_row_end, k_col_start:k_col_end]

            kernel_flat = kernel.reshape((img.shape[0], kernel_size**2))

            abs_max_idx = np.argmax(np.abs(kernel_flat), axis=1)

            img[:, row, col] = kernel_flat[np.arange(kernel_flat.shape[0]), abs_max_idx]
    return img

@keras.utils.register_keras_serializable()
class CylindricalPadding2D(keras.layers.Layer):
    """
    Cylindrical colvolution: https://stackoverflow.com/questions/54911015/keras-convolution-layer-on-images-coming-from-circular-cyclic-domain
    """

    def __init__(self, offset, axis=2, input_dim=32):
        super().__init__()
        self.offset = tf.constant(offset)
        self.axis = tf.constant(axis)
    
    def get_config(self):
        config = super().get_config()
        config["offset"] = int(self.offset)
        return config

    def call(self, inputs):
        extra_right = inputs[:, :, -self.offset:, :]
        extra_left = inputs[:, :, :self.offset, :]
        return tf.concat([extra_right, inputs, extra_left], axis=self.axis)


def test_cylinder():
    input_img = keras.Input(shape=[3,12,1])

    x = input_img
    x = CylindricalPadding2D(3)(x)

    model = keras.Model(input_img, x)

    test = np.array([range(12),range(0,24,2),range(0,36,3)]).reshape((1,3,12,1))
    result = model(test)[0,:,:,0]

    print("\n- Original")
    print(np.asarray(test[0,:,:,0]).astype(int))

    print("\n- Cylindical Padding x3")
    print(np.asarray(result).astype(int))

def test_polar_linear():
    import matplotlib.pyplot as plt

    # zebra = np.tile([1,0], 50*360).reshape([100,360]).T

    zebra = np.tile([1,0], 50*100).reshape([100,100]).T
    zebra = zebra * np.linspace(0,1,100)
    plt.imshow(zebra, cmap="Greys")
    plt.show()

    # circle_zebra = polar_to_linear(zebra)
    # plt.imshow(circle_zebra, cmap="Greys")
    # plt.show()

    zebra = np.tile([1,0], 50*100).reshape([100,100]).T
    zebra = zebra * np.linspace(0,1,100)
    zebra = polar_to_linear(zebra)

    # circle_zebra = polar_to_linear_tf(zebra, 100)
    circle_zebra = linear_to_polar_tf(zebra, 100)
    plt.imshow(circle_zebra, cmap="Greys")
    plt.show()


if __name__ == "__main__":
    test_cylinder()
    # test_polar_linear()
