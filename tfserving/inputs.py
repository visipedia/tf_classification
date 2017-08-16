"""
Numpy and scipy image preparation.

Author: Grant Van Horn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imresize

def prepare_image(image, input_height=299, input_width=299):
  """ Prepare an image to be passed through a network.
  Arguments:
    image (numpy.ndarray): An uint8 RGB image
  Returns:
    list: the image resized, centered and raveled
  """

  # We assume an uint8 RGB image
  assert image.dtype == np.uint8
  assert image.ndim == 3
  assert image.shape[2] == 3

  resized_image = imresize(image, (input_height, input_width, 3))
  float_image = resized_image.astype(np.float32)
  centered_image = ((float_image / 255.) - 0.5) * 2.0

  return centered_image.ravel().tolist()
