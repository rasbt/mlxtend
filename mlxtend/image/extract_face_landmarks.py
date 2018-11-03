# Sebastian Raschka 2014-2018
# contributor: Vahid Mirjalili
# mlxtend Machine Learning Library Extensions
#
# A counter class for printing the progress of an iterator.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import dlib
import os
from .utils import makedir, check_exists, download_url, extract_file

predictor_path = '~/mlxtend_data/shape_predictor_68_face_landmarks.dat'
predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

if not check_exists(predictor_path):
    download_url(predictor_url, save_path='~/mlxtend_data/', 
                 filename='shape_predictor_68_face_landmarks.dat.bz2')
    extract_file('~/mlxtend_data/shape_predictor_68_face_landmarks.dat.bz2')
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def extract_face_landmarks():
    pass    
