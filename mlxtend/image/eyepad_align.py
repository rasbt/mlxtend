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
import pyprind


left_indx = np.array([36, 37, 38, 39, 40, 41])
right_indx = np.array([42, 43, 44, 45, 46, 47])


class EyepadAlign(object):
    """Class to align/transform face images to target landmarks,
       based on the location of the eyes.

       1. Scaling factor is computed based on distance between the
        left and right eyes, so that the transformed image will
        have the same eye distance as target.

       2. Transformation is performed based on the eyes' middle point.

       3. Finally, the transformed image is padded with zeros to match
        the desired final image size.

    Parameters
    ----------

    target_landmarks : target landmarks to transform new face images to

    target_width : the width of the output image

    target_height : the height of the output image

    verbose : verbose level to display the progress bar and log messages

    Attributes
    ----------

    target_landmarks_ : target landmarks to transform new face images to,
        which can be either (1) assigned to pre-fit shapes,
                            (2) can be computed from a single face image
                            (3) can be cmputed as the mean of face landmarks
                                from all face images in a directory.

    target_width_ : the width of the transformed output image.

    target_height_ : the height of the transformed output image.


    Examples
    --------
        eyepad = EyepadAlign()
        eyepad.fit(target_image=img_a)
        img_tr = eyepad.transform(img_b)

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/image/EyepadAlign/

    """
    def __init__(self, target_landmarks=None, taregt_width=None,
                 target_height=None, verbose=0):
        self.eye_distance = None
        self.target_landmarks_ = target_landmarks
        self.taregt_width_ = taregt_width
        self.target_height_ = target_height
        self.verbose = verbose

    def fit(self, target_image=None,
            target_img_dir=None, file_extensions='.jpg'):
        """Fits the target landmarks points:
             a. if target_image is given, sets the target landmarks
                to the landmarks of target image.
             b. otherwise, if a target directory is given,
                calculates the average landmarks for all face images
                in the directory which will be set as the target landmark.

        """
        if target_image is not None:
            landmarks = extract_face_landmarks(target_image)
            self.target_landmarks_ = landmarks
            self.target_width_ = target_image.shape[1]
            self.target_height_ = target_image.shape[0]

        elif target_img_dir is not None:
            file_list = listdir(target_img_dir, file_extensions)
            if self.verbose >= 1:
                print("Fitting the average facial landmarks "
                      "for {} face images ".format(len(file_list)))
            landmarks_list = []
            pbar = pyprind.ProgBar(len(file_list))
            for f in file_list:
                pbar.update()
                img = read_image(filename=f, path=target_img_dir)
                landmarks = extract_face_landmarks(img)
                if landmarks is not None:
                    landmarks_list.append(landmarks)
            self.target_landmarks_ = np.mean(landmarks_list, axis=0)
            self.target_width_ = img.shape[1]
            self.target_height_ = img.shape[0]

    def _cal_eye_properties(self, landmarks):
        """ Calculates the face properties:
               (1) coordinates of the left-eye
               (2) coordinates of the right-eye
               (3) the distance between left and right eyes
               (4) the middle point between the two eyes
        """
        left_eye = np.mean(landmarks[left_indx], axis=0)
        right_eye = np.mean(landmarks[right_indx], axis=0)
        eyes_mid_point = (left_eye + right_eye)/2.0
        eye_distance = np.sqrt(np.sum(np.square(left_eye - right_eye)))

        return eyes_mid_point, eye_distance

    def transform(self, img):
        """ transforms a single face image (img) to the target landmarks
               based on the location of the eyes by
               scaling, translation and cropping (if needed):

            (1) Scaling the image so that the distance of the two eyes
                in the given image (img) matches the distance of the
                two eyes in the target landmarks.

            (2) Translation is performed based on the middle point
                between the two eyes.
        """
        if self.eye_distance is None:
            props = self._cal_eye_properties(self.target_landmarks_)
            self.eyes_mid_point = props[0]
            self.eye_distance = props[1]
        landmarks = extract_face_landmarks(img)
        if landmarks is None:
            return
        eyes_mid_point, eye_distance = self._cal_eye_properties(landmarks)

        scale = self.eye_distance / eye_distance
        tr = (self.eyes_mid_point/scale - eyes_mid_point)
        tr = (int(tr[0]*scale), int(tr[1]*scale))

        tform = AffineTransform(scale=(scale, scale), rotation=0, shear=0,
                                translation=tr)
        h, w = self.target_height_, self.target_width_
        img_tr = warp(img, tform.inverse, output_shape=(h, w))
        return np.array(img_tr*255, dtype='uint8')
