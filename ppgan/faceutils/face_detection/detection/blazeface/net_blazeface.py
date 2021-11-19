# code was heavily based on https://github.com/hollance/BlazeFace-PyTorch
# This work is licensed under the same terms as MediaPipe (Apache License 2.0)
# https://github.com/google/mediapipe/blob/master/LICENSE

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BlazeBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2D(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2D(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=in_channels),
            nn.Conv2D(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, [0, 2, 0, 2], "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, [0, 0, 0, self.channel_pad, 0, 0, 0, 0], "constant", 0)

        return self.act(self.convs(h) + x)


class BlazeFace(nn.Layer):
    """The BlazeFace face detection model.
    """
    def __init__(self):
        super(BlazeFace, self).__init__()

        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2D(in_channels=3,
                      out_channels=24,
                      kernel_size=5,
                      stride=2,
                      padding=0),
            nn.ReLU(),
            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )

        self.classifier_8 = nn.Conv2D(88, 2, 1)
        self.classifier_16 = nn.Conv2D(96, 6, 1)

        self.regressor_8 = nn.Conv2D(88, 32, 1)
        self.regressor_16 = nn.Conv2D(96, 96, 1)

    def forward(self, x):
        x = F.pad(x, [1, 2, 1, 2], "constant", 0)

        b = x.shape[0]

        x = self.backbone1(x)  # (b, 88, 16, 16)
        h = self.backbone2(x)  # (b, 96, 8, 8)

        c1 = self.classifier_8(x)  # (b, 2, 16, 16)
        c1 = c1.transpose([0, 2, 3, 1])  # (b, 16, 16, 2)
        c1 = c1.reshape([b, -1, 1])  # (b, 512, 1)

        c2 = self.classifier_16(h)  # (b, 6, 8, 8)
        c2 = c2.transpose([0, 2, 3, 1])  # (b, 8, 8, 6)
        c2 = c2.reshape([b, -1, 1])  # (b, 384, 1)

        c = paddle.concat((c1, c2), axis=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)  # (b, 32, 16, 16)
        r1 = r1.transpose([0, 2, 3, 1])  # (b, 16, 16, 32)
        r1 = r1.reshape([b, -1, 16])  # (b, 512, 16)

        r2 = self.regressor_16(h)  # (b, 96, 8, 8)
        r2 = r2.transpose([0, 2, 3, 1])  # (b, 8, 8, 96)
        r2 = r2.reshape([b, -1, 16])  # (b, 384, 16)

        r = paddle.concat((r1, r2), axis=1)  # (b, 896, 16)
        return [r, c]

    def load_weights(self, path):
        paddle.load_dict(paddle.load(path))
        self.eval()

    def load_anchors(self, path):
        self.anchors = paddle.to_tensor(np.load(path), dtype='float32')
        assert (self.anchors.shape == 2)
        assert (self.anchors.shape[0] == self.num_anchors)
        assert (self.anchors.shape[1] == 4)

    def load_anchors_from_npy(self, arr):
        self.anchors = paddle.to_tensor(arr, dtype='float32')
        assert (len(self.anchors.shape) == 2)
        assert (self.anchors.shape[0] == self.num_anchors)
        assert (self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.astype('float32') / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a Paddle tensor of
                 shape (3, H, W). The image's height and width should be
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = paddle.to_tensor(img).transpose((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a Paddle tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a Paddle tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = paddle.to_tensor(x).transpose((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 128
        assert x.shape[3] == 128

        x = self._preprocess(x)

        with paddle.no_grad():
            out = self.__call__(x)

        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = paddle.stack(faces) if len(faces) > 0 else paddle.zeros(
                (0, 17))
            filtered_detections.append(faces)

        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.

        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.
        """
        assert len(raw_box_tensor.shape) == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert len(raw_score_tensor.shape) == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clip(-thresh, thresh)
        detection_scores = F.sigmoid(raw_score_tensor).squeeze(axis=-1)

        mask = detection_scores >= self.min_score_thresh
        mask = mask.numpy()
        detection_boxes = detection_boxes.numpy()
        detection_scores = detection_scores.numpy()

        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = paddle.to_tensor(detection_boxes[i, mask[i]])
            scores = paddle.to_tensor(
                detection_scores[i, mask[i]]).unsqueeze(axis=-1)
            output_detections.append(paddle.concat((boxes, scores), axis=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = paddle.zeros_like(raw_boxes)

        x_center = raw_boxes[:,:, 0] / self.x_scale * \
            anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[:,:, 1] / self.y_scale * \
            anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[:, :, 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[:, :, 3] / self.h_scale * anchors[:, 3]

        boxes[:, :, 0] = y_center - h / 2.  # ymin
        boxes[:, :, 1] = x_center - w / 2.  # xmin
        boxes[:, :, 2] = y_center + h / 2.  # ymax
        boxes[:, :, 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[:,:, offset] / \
                self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[:,:, offset + 1] / \
                self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[:, :, offset] = keypoint_x
            boxes[:, :, offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:
        The input detections should be a Tensor of shape (count, 17).
        Returns a list of Paddle tensors, one for each detected face.

        """
        if len(detections) == 0:
            return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = paddle.argsort(detections[:, 16], descending=True).numpy()
        detections = detections.numpy()

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(paddle.to_tensor(first_box),
                                      paddle.to_tensor(other_boxes))

            mask = ious > self.min_suppression_threshold
            mask = mask.numpy()

            overlapping = remaining[mask]
            remaining = remaining[~mask]

            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(axis=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(paddle.to_tensor(weighted_detection))

        return output_detections


def intersect(box_a, box_b):
    """Compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = paddle.minimum(box_a[:, 2:].unsqueeze(1).expand((A, B, 2)),
                            box_b[:, 2:].unsqueeze(0).expand((A, B, 2)))
    min_xy = paddle.maximum(box_a[:, :2].unsqueeze(1).expand((A, B, 2)),
                            box_b[:, :2].unsqueeze(0).expand((A, B, 2)))
    inter = paddle.clip((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


def init_model():
    net = BlazeFace()
    net.load_weights("blazeface.pdparams")
    net.load_anchors("anchors.npy")

    net.min_score_thresh = 0.75
    net.min_suppression_threshold = 0.3

    return net
