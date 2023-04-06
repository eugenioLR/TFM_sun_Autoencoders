import numpy as np

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


def polar_to_linear(img, o=None, r=None, output=None, order=1, cont=0, cval=0):
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


# def polar_to_linear_tf(img_input, radius):
#     h, w = radius, radius

#     r, theta = tf.meshgrid(tf.linspace(0.0, radius, w), tf.linspace(0.0, 2.0*np.pi, h))

#     x, y = tf.meshgrid(tf.linspace(0.0, radius, w), tf.linspace(0.0, radius, h))

#     r = tf.cast(r, tf.float64)
#     theta = tf.cast(theta, tf.float64)
#     x = tf.cast(x, tf.float64)
#     y = tf.cast(y, tf.float64)

#     src_coords = tf.reshape(tf.stack([r, theta], axis=-1), [1, 2, -1])
#     dst_coords = tf.reshape(tf.stack([x, y], axis=-1), [1, 2, -1])

#     print(src_coords)
#     print(dst_coords)

#     interpolated = tfa.image.sparse_image_warp(img_input, src_coords, dst_coords) #?

#     return interpolated


class CylindricalPadding2D(keras.layers.Layer):
    """
    Cylindrical colvolution: https://stackoverflow.com/questions/54911015/keras-convolution-layer-on-images-coming-from-circular-cyclic-domain
    """

    def __init__(self, offset, axis=2, input_dim=32):
        super().__init__()
        self.offset = tf.constant(offset)
        self.axis = tf.constant(axis)

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

    zebra = np.tile([1,0], 50*360).reshape([100,360]).T
    plt.imshow(zebra, cmap="Greys")
    plt.show()

    circle_zebra = polar_to_linear(zebra)
    plt.imshow(circle_zebra, cmap="Greys")
    plt.show()

if __name__ == "__main__":
    test_cylinder()
    # test_polar_linear()
