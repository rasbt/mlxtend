# Sebastian Raschka 2014-2018
# contributor: Vahid Mirjalili
# mlxtend Machine Learning Library Extensions
#
# A class for transforming face images.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from . import extract_face_landmarks
from .utils import listdir, read_image
from skimage.transform import warp, AffineTransform
import numpy as np

left_indx = np.array([36, 37, 38, 39, 40, 41])
right_indx = np.array([42, 43, 44, 45, 46, 47])


class EyepadAlign():
    def __init__(self, target_landmarks=None,
                 image_width=None, image_height=None):
        self.target_landmarks = target_landmarks
        self.target_width = image_width
        self.target_height = image_height

        self.eyes_mid_point = None
        self.eyes_distance = None

    def fit(self, target_image=None,
            target_img_dir=None, file_extensions='.jpg'):
        if target_image is not None:
            landmarks = extract_face_landmarks(target_image)
            if landmarks is not None:
                self.target_landmarks = landmarks
            self.target_width = target_image.shape[0]
            self.target_height = target_image.shape[1]
        elif target_img_dir is not None:
            file_list = listdir(target_img_dir, file_extensions)
            print("Fitting the average facial landmarks "
                  "for {} face images ".format(len(file_list)))
            landmarks_list = []
            for f in file_list:
                img = read_image(filename=f, path=target_img_dir)
                landmarks = extract_face_landmarks(img)
                if landmarks is not None:
                    landmarks_list.append(landmarks)
            self.target_landmarks = np.mean(landmarks_list, axis=0)
            self.target_width = img.shape[0]
            self.target_height = img.shape[1]

    def cal_eye_properties(self, landmarks):
        left_eye = np.mean(landmarks[left_indx], axis=0)
        right_eye = np.mean(landmarks[right_indx], axis=0)
        eyes_mid_point = (left_eye + right_eye)/2.0
        eyes_distance = np.sqrt(np.sum(np.square(left_eye - right_eye)))

        return eyes_mid_point, eyes_distance

    def transform(self, img):
        if self.eyes_distance is None:
            props = self.cal_eye_properties(self.target_landmarks)
            self.eyes_mid_point = props[0]
            self.eyes_distance = props[1]
        landmarks = extract_face_landmarks(img)
        if landmarks is None:
            return
        eyes_mid_point, eyes_distance = self.cal_eye_properties(landmarks)

        scale = self.eyes_distance / eyes_distance
        tr = (self.eyes_mid_point/scale - eyes_mid_point)
        tr = (int(tr[0]*scale), int(tr[1]*scale))

        tform = AffineTransform(scale=(scale, scale), rotation=0, shear=0,
                                translation=tr)
        h, w = self.target_height, self.target_width
        img_tr = warp(img, tform.inverse, output_shape=(h, w))
        return np.array(img_tr*255, dtype='uint8')
