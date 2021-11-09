import numpy as np
import face_alignment as fa
import cv2

def align_face(image, left_eye, right_eye):
  
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0] 
    left_eye_y = left_eye_center[1]
    
    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]
    # if left_eye_y > right_eye_y:
    #     A = (right_eye_x, left_eye_y)
    #     direction = -1 
    # else:
    #     A = (left_eye_x, right_eye_y)
    #     direction = 1 
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle=np.arctan(delta_y/(delta_x+1))
    angle = (angle * 180) / np.pi
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
    
def get_eyes(fa_predictor, image):
    landmarks = fa_predictor.get_landmarks(image)[0]
    left_eye = np.array(landmarks[36:42])
    right_eye = np.array(landmarks[42:48])
    left_eye = np.array([min(left_eye[:, 0]), min(left_eye[:,1]), 
                        max(left_eye[:, 0]), max(left_eye[:,1])]).astype("int")
    right_eye = np.array([min(right_eye[:,0]), min(right_eye[:,1]), 
                        max(right_eye[:,0]), max(right_eye[:,1])]).astype("int")
    
    return left_eye, right_eye

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

