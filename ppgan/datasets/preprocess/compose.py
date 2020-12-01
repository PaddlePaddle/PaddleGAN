# from collections.abc import Sequence

# from mmcv.utils import build_from_cfg

# class Compose(object):
#     """Compose a data pipeline with a sequence of transforms.

#     Args:
#         transforms (list[dict | callable]):
#             Either config dicts of transforms or transform objects.
#     """

#     def __init__(self, transforms):
#         assert isinstance(transforms, Sequence)
#         self.transforms = []
#         for transform in transforms:
#             if isinstance(transform, dict):
#                 transform = build_from_cfg(transform, PIPELINES)
#                 self.transforms.append(transform)
#             elif callable(transform):
#                 self.transforms.append(transform)
#             else:
#                 raise TypeError(f'transform must be callable or a dict, '
#                                 f'but got {type(transform)}')

#     def __call__(self, data):
#         """Call function.

#         Args:
#             data (dict): A dict containing the necessary information and
#                 data for augmentation.

#         Returns:
#             dict: A dict containing the processed data and information.
#         """
#         for t in self.transforms:
#             data = t(data)
#             if data is None:
#                 return None
#         return data
