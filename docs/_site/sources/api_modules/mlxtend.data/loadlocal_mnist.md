## loadlocal_mnist

*loadlocal_mnist(images_path, labels_path)*

Read MNIST from ubyte files.

**Parameters**

- `images_path` : str

    path to the test or train MNIST ubyte file

- `labels_path` : str

    path to the test or train MNIST class labels file

**Returns**

- `images` : [n_samples, n_pixels] numpy.array

    Pixel values of the images.

- `labels` : [n_samples] numpy array

    Target class labels

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/](http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/)

