## extract_face_landmarks

*extract_face_landmarks(img, return_dtype=<class 'numpy.int32'>)*

Function to extract face landmarks.

Note that this function requires an installation of
the Python version of the library "dlib": http://dlib.net

**Parameters**

- `img` : array, shape = [h, w, ?]

    numpy array of a face image.
    Supported shapes are
    - 3D tensors with 1
    or more color channels, for example,
    RGB: [h, w, 3]
    - 2D tensors without color channel, for example,
    Grayscale: [h, w]
    return_dtype: the return data-type of the array,
    default: np.int32.

**Returns**

- `landmarks` : numpy.ndarray, shape = [68, 2]

    A numpy array, where each row contains a landmark/point x-y coordinates.
    Return None if no face is detected by Dlib.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/sources/image/extract_face_landmarks.ipynb](http://rasbt.github.io/mlxtend/user_guide/sources/image/extract_face_landmarks.ipynb)

