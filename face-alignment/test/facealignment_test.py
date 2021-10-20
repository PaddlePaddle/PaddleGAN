import unittest
import numpy as np
import face_alignment
import sys
import torch
sys.path.append('.')
from face_alignment.utils import get_image


class Tester(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_data = [np.array([[137., 240., -85.907196],
                                         [140., 264., -81.1443],
                                         [143., 288., -76.25633],
                                         [146., 306., -69.01708],
                                         [152., 327., -53.775352],
                                         [161., 342., -30.029667],
                                         [170., 348., -2.792292],
                                         [185., 354., 23.522688],
                                         [212., 360., 38.664257],
                                         [239., 357., 31.747217],
                                         [263., 354., 12.192401],
                                         [284., 348., -10.0569725],
                                         [302., 333., -29.42916],
                                         [314., 315., -41.675602],
                                         [320., 297., -46.924263],
                                         [326., 276., -50.33218],
                                         [335., 252., -53.945686],
                                         [152., 207., -7.6189857],
                                         [164., 201., 6.1879144],
                                         [176., 198., 16.991247],
                                         [188., 198., 24.690582],
                                         [200., 201., 29.248188],
                                         [245., 204., 37.878166],
                                         [257., 201., 37.420483],
                                         [269., 201., 34.163113],
                                         [284., 204., 28.480812],
                                         [299., 216., 18.31863],
                                         [221., 225., 37.93351],
                                         [218., 237., 48.337395],
                                         [215., 249., 60.502884],
                                         [215., 261., 63.353687],
                                         [203., 273., 40.186855],
                                         [209., 276., 45.057003],
                                         [218., 276., 48.56715],
                                         [227., 276., 47.744766],
                                         [233., 276., 45.01401],
                                         [170., 228., 7.166072],
                                         [179., 222., 17.168053],
                                         [188., 222., 19.775822],
                                         [200., 228., 19.06176],
                                         [191., 231., 20.636724],
                                         [179., 231., 16.125824],
                                         [248., 231., 28.566122],
                                         [257., 225., 33.024036],
                                         [269., 225., 34.384735],
                                         [278., 231., 27.014532],
                                         [269., 234., 32.867023],
                                         [257., 234., 33.34033],
                                         [185., 306., 29.927242],
                                         [194., 297., 42.611233],
                                         [209., 291., 50.563396],
                                         [215., 291., 52.831104],
                                         [221., 291., 52.9225],
                                         [236., 300., 48.32575],
                                         [248., 309., 38.2375],
                                         [236., 312., 48.377922],
                                         [224., 315., 52.63793],
                                         [212., 315., 52.330444],
                                         [203., 315., 49.552994],
                                         [194., 309., 42.64459],
                                         [188., 303., 30.746407],
                                         [206., 300., 46.514435],
                                         [215., 300., 49.611156],
                                         [224., 300., 49.058918],
                                         [248., 309., 38.084103],
                                         [224., 303., 49.817806],
                                         [215., 303., 49.59815],
                                         [206., 303., 47.13894]], dtype=np.float32)]

    def test_predict_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu')
        preds = fa.get_landmarks('test/assets/aflw-test.jpg')
        self.assertEqual(len(preds), len(self.reference_data))
        for pred, reference in zip(preds, self.reference_data):
            self.assertTrue(np.allclose(pred, reference))

    def test_predict_batch_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu')

        reference_data = self.reference_data + self.reference_data
        reference_data.append([])
        image = get_image('test/assets/aflw-test.jpg')
        batch = np.stack([image, image, np.zeros_like(image)])
        batch = torch.Tensor(batch.transpose(0, 3, 1, 2))

        preds = fa.get_landmarks_from_batch(batch)

        self.assertEqual(len(preds), len(reference_data))
        for pred, reference in zip(preds, reference_data):
            self.assertTrue(np.allclose(pred, reference))

    def test_predict_points_from_dir(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu')

        reference_data = {
            'test/assets/grass.jpg': None,
            'test/assets/aflw-test.jpg': self.reference_data}

        preds = fa.get_landmarks_from_directory('test/assests/')

        for k, points in preds.items():
            if isinstance(points, list):
                for p, p_reference in zip(points, reference_data[k]):
                    self.assertTrue(np.allclose(p, p_reference))
            else:
                self.assertEqual(points, reference_data[k])


if __name__ == '__main__':
    unittest.main()
