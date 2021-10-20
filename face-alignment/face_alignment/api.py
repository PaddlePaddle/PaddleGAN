import torch
import warnings
from enum import IntEnum
from skimage import io
import numpy as np
from distutils.version import LooseVersion

from .utils import *


class LandmarksType(IntEnum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(IntEnum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4


default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

models_urls = {
    '1.6': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip',
    },
    '1.5': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip',
    },
}


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', face_detector_kwargs=None, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        if LooseVersion(torch.__version__) < LooseVersion('1.5.0'):
            raise ImportError(f'Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
                            Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0')

        network_size = int(network_size)
        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        face_detector_kwargs = face_detector_kwargs or {}
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)

        # Initialise the face alignemnt networks
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)
        self.face_alignment_net = torch.jit.load(
            load_file_from_url(models_urls.get(pytorch_version, default_model_urls)[network_name]))

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = torch.jit.load(
                load_file_from_url(models_urls.get(pytorch_version, default_model_urls)['depth']))

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None, return_bboxes=False, return_landmark_score=False):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces, return_bboxes, return_landmark_score)

    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """
        image = get_image(image_or_path)

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image.copy())

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores = []
        for i, d in enumerate(detected_faces):
            center = torch.tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment_net(inp).detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
            out = out.cpu().numpy()

            pts, pts_img, scores = get_preds_fromhm(out, center.numpy(), scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            scores = scores.squeeze(0)

            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0 and pts[i, 1] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device)
                depth_pred = self.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())
            landmarks_scores.append(scores)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores, detected_faces
        else:
            return landmarks

    @torch.no_grad()
    def get_landmarks_from_batch(self, image_batch, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_batch {torch.tensor} -- The input images batch

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmarks
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_batch(image_batch)

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores_list = []
        # A batch for each frame
        for i, faces in enumerate(detected_faces):
            res = self.get_landmarks_from_image(
                image_batch[i].cpu().numpy().transpose(1, 2, 0),
                detected_faces=faces,
                return_landmark_score=return_landmark_score,
            )
            if return_landmark_score:
                landmark_set, landmarks_scores, _ = res
                landmarks_scores_list.append(landmarks_scores)
            else:
                landmark_set = res
            # Bacward compatibility
            if landmark_set is not None:
                landmark_set = np.concatenate(landmark_set, axis=0)
            else:
                landmark_set = []
            landmarks.append(landmark_set)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores_list = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores_list, detected_faces
        else:
            return landmarks

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True,
                                     return_bboxes=False, return_landmark_score=False):
        """Scan a directory for images with a given extension type(s) and predict the landmarks for each
            face present in the images found.

         Arguments:
            path {str} -- path to the target directory containing the images

        Keyword Arguments:
            extensions {list of str} -- list containing the image extensions considered (default: ['.jpg', '.png'])
            recursive {boolean} -- If True, scans for images recursively (default: True)
            show_progress_bar {boolean} -- If True displays a progress bar (default: True)
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        """
        detected_faces = self.face_detector.detect_from_directory(path, extensions, recursive, show_progress_bar)

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            if return_bboxes or return_landmark_score:
                preds, bbox, score = self.get_landmarks_from_image(
                    image, bounding_boxes, return_bboxes=return_bboxes, return_landmark_score=return_landmark_score)
                predictions[image_path] = (preds, bbox, score)
            else:
                preds = self.get_landmarks_from_image(image, bounding_boxes)
                predictions[image_path] = preds

        return predictions
