mlxtend version: 0.15.0dev 
## EyepadAlign

*EyepadAlign(verbose=0)*

Class to align/transform face images to facial landmarks,
based on eye alignment.

1. A scaling factor is computed based on distance between the
left and right eye, such that the transformed face image will
have the same eye distance as a reference face image.

2. A transformation is performed based on the eyes' center point.
to align the face based on the reference eye location.

3. Finally, the transformed image is padded with zeros to match
the desired final image size.

**Parameters**

- `verbose` : int (default=0)

    Verbose level to display the progress bar and log messages.
    Setting `verbose=1` will print a progress bar upon calling
    `fit_directory`.

**Attributes**

- `target_landmarks_` : target landmarks to transform new face images to.

    Depending on the chosen `fit` parameters, it can  be either
    (1) assigned to pre-fit shapes,
    (2) computed from a single face image
    (3) computed as the mean of face landmarks
    from all face images in a file directory of face images.


- `eye_distance_` : the distance between left and right eyes

    in the target landmarks.


- `target_height_` : the height of the transformed output image.



- `target_width_` : the width of the transformed output image.


For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/image/EyepadAlign/](http://rasbt.github.io/mlxtend/user_guide/image/EyepadAlign/)

**Returns**

- `self` : object


### Methods

<hr>

*fit_directory(target_img_dir, target_height, target_width, file_extension='.jpg', pre_check=True)*

Calculates the average landmarks for all face images
in a directory which will then be set as the target landmark set.

**Arguments**

- `target_img_dir` : str

    Directory containing the images


- `target_height` : int

    Expected image height of the images in the directory


- `target_width` : int

    Expected image width of the images in the directory

    file_extension str (default='.jpg'): File extension of the image files.

    pre_check Bool (default=True): Checks that each image has the dimensions
    specificed via `target_height` and `target_width` on the whole
    directory first to identify potential issues that are recommended
    to be fixed before proceeding. Raises a warning for each image if
    dimensions differ from the ones specified and expected.

**Returns**

- `self` : object


<hr>

*fit_image(target_image)*

Derives facial landmarks from a target image.

**Arguments**

- `target_image` : `uint8` numpy.array, shape=[width, height, channels]

    NumPy array representation of the image data.

**Returns**

- `self` : object


<hr>

*fit_values(target_landmarks, target_width, target_height)*

Used for determining the eye location from pre-defined
landmark arrays, eliminating the need for re-computing
the average landmarks on a target image or image directory.

**Arguments**

- `target_landmarks` : np.array, shape=(height, width)

    NumPy array containing the locations of the facial landmarks
    as determined by `mlxtend.image.extract_face_landmarks`


- `target_height` : int

    image height


- `target_width` : int

    image width

**Returns**

- `self` : object


<hr>

*transform(img)*

transforms a single face image (img) to the target landmarks
based on the location of the eyes by
scaling, translation and cropping (if needed):

(1) Scaling the image so that the distance of the two eyes
in the given image (img) matches the distance of the
two eyes in the target landmarks.

(2) Translation is performed based on the middle point
between the two eyes.

**Arguments**


- `img` : np.array, shape=(height, width, channels)

    Input image to be transformed.

**Returns**

- `self` : object





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




