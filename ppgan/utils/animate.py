import numpy as np
from scipy.spatial import ConvexHull

import paddle


def normalize_kp(kp_source,
                 kp_driving,
                 kp_driving_initial,
                 adapt_movement_scale=False,
                 use_relative_movement=False,
                 use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = paddle.matmul(
                kp_driving['jacobian'],
                paddle.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = paddle.matmul(jacobian_diff,
                                               kp_source['jacobian'])

    return kp_new
