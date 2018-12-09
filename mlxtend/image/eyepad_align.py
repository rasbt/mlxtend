# Sebastian Raschka 2014-2018
# contributor: Vahid Mirjalili
# mlxtend Machine Learning Library Extensions
#
# A class for transforming face images.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from . import extract_face_landmarks
from .utils import listdir, read_image
from ..externals.pyprind.progbar import ProgBar
from skimage.transform import warp, AffineTransform, resize


LEFT_INDEX = np.array([36, 37, 38, 39, 40, 41])
RIGHT_INDEX = np.array([42, 43, 44, 45, 46, 47])


class EyepadAlign(object):
    """Class to align/transform face images to facial landmarks,
       based on eye alignment.

       1. A scaling factor is computed based on distance between the
        left and right eye, such that the transformed face image will
        have the same eye distance as a reference face image.

       2. A transformation is performed based on the eyes' center point.
        to align the face based on the reference eye location.

       3. Finally, the transformed image is padded with zeros to match
        the desired final image size.

    Parameters
    ----------
    verbose : int (default=0)
        Verbose level to display the progress bar and log messages.
        Setting `verbose=1` will print a progress bar upon calling
        `fit_directory`.

    Attributes
    ----------
    target_landmarks_ : target landmarks to transform new face images to.
        Depending on the chosen `fit` parameters, it can  be either
          (1) assigned to pre-fit shapes,
          (2) computed from a single face image
          (3) computed as the mean of face landmarks
              from all face images in a file directory of face images.

    eye_distance_ : the distance between left and right eyes
        in the target landmarks.

    target_width_ : the width of the transformed output image.

    target_height_ : the height of the transformed output image.

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/image/EyepadAlign/

    Return
    ---------------
    self : object

    """
    def __init__(self, verbose=0):
        self.verbose = verbose

    def fit_image(self, target_image):
        """if target_image is given, sets the target landmarks
                to the landmarks of target image.

        """
        landmarks = extract_face_landmarks(target_image)
        self.target_landmarks_ = landmarks
        self.target_width_ = target_image.shape[1]
        self.target_height_ = target_image.shape[0]

        props = self._calc_eye_properties(self.target_landmarks_)
        self.eyes_mid_point_ = props[0]
        self.eye_distance_ = props[1]
        return self

    def fit_directory(self, target_img_dir,
                      target_width, target_height,
                      file_extensions='.jpg'):
        """
        calculates the average landmarks for all face images
        in the directory which will be set as the target landmark.

        """
        self.target_width_ = target_width
        self.target_height_ = target_height

        file_list = listdir(target_img_dir, file_extensions)
        if self.verbose >= 1:
            print("Fitting the average facial landmarks "
                  "for %d face images " % (len(file_list)))
        landmarks_list = []

        if self.verbose >= 1:
            pbar = ProgBar(len(file_list))
        for f in file_list:
            if self.verbose >= 1:
                pbar.update()
            img = read_image(filename=f, path=target_img_dir)

            if self.target_width_ != img.shape[1]:
                width_ratio = self.target_width_ / img.shape[1]
                height_ratio = self.target_height_ / img.shape[0]

                if np.abs(width_ratio - height_ratio) > 0.001:  # ignore
                    continue

                img = resize(img, output_shape=(self.target_height_,
                                                self.target_width_),
                             anti_aliasing=True, mode='reflect')
                img = (img*255).astype('uint8')

            landmarks = extract_face_landmarks(img)
            if np.sum(landmarks) is not None:  # i.e., None == no face detected
                landmarks_list.append(landmarks)
        self.target_landmarks_ = np.mean(landmarks_list, axis=0)

        props = self._calc_eye_properties(self.target_landmarks_)
        self.eyes_mid_point_ = props[0]
        self.eye_distance_ = props[1]
        return self

    def fit_values(self, target_landmarks, target_width, target_height):
        """ sets the values for target_landmarks, target_width,
               and target_heigh.
            It can be used for assigning these arrays/values to
               pre-fit models, elliminating the need for re-computing
               the average landmarks on a target image-dir.
        """
        self.target_landmarks_ = target_landmarks
        self.target_width_ = target_width
        self.target_height_ = target_height

        props = self._calc_eye_properties(self.target_landmarks_)
        self.eyes_mid_point_ = props[0]
        self.eye_distance_ = props[1]
        return self

    def _calc_eye_properties(self, landmarks):
        """ Calculates the face properties:
               (1) coordinates of the left-eye
               (2) coordinates of the right-eye
               (3) the distance between left and right eyes
               (4) the middle point between the two eyes
        """
        left_eye = np.mean(landmarks[LEFT_INDEX], axis=0)
        right_eye = np.mean(landmarks[RIGHT_INDEX], axis=0)
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

        ## ADD CHECK TO MAKE SURE FIT* was called

        landmarks = extract_face_landmarks(img)
        if landmarks is None:
            return

        eyes_mid_point, eye_distance = self._calc_eye_properties(landmarks)

        scale = self.eye_distance_ / eye_distance
        tr = (self.eyes_mid_point_/scale - eyes_mid_point)
        tr = (int(tr[0]*scale), int(tr[1]*scale))

        tform = AffineTransform(scale=(scale, scale), rotation=0, shear=0,
                                translation=tr)
        h, w = self.target_height_, self.target_width_
        img_tr = warp(img, tform.inverse, output_shape=(h, w))
        return np.array(img_tr*255, dtype='uint8')
