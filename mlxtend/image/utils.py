# Sebastian Raschka 2014-2019
# contributor: Vahid Mirjalili
# mlxtend Machine Learning Library Extensions
#
# A counter class for printing the progress of an iterator.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os
import tarfile
import zipfile
import bz2
import imageio


def check_exists(path):
    path = os.path.expanduser(path)
    return os.path.exists(path)


def makedir(path):
    path = os.path.expanduser(path)
    if not check_exists(path):
        os.makedirs(path)


def listdir(path, extensions=''):
    path = os.path.expanduser(path)
    if check_exists(path):
        return [f for f in os.listdir(path) if f.endswith(extensions)]
    else:
        raise FileNotFoundError


def read_image(filename, path=None):
    if path is not None:
        path = os.path.expanduser(path)
        filename = os.path.join(path, filename)
    if check_exists(filename):
        return imageio.imread(filename)
    else:
        raise FileNotFoundError


def download_url(url, save_path):
    from six.moves import urllib
    save_path = os.path.expanduser(save_path)
    if not check_exists(save_path):
        makedir(save_path)

    filename = url.rpartition('/')[2]
    filepath = os.path.join(save_path, filename)

    try:
        print('Downloading '+url+' to '+filepath)
        urllib.request.urlretrieve(url, filepath)
    except ValueError:
        raise Exception('Failed to download! Check URL: ' + url +
                        ' and local path: ' + save_path)


def extract_file(path, to_directory=None):
    path = os.path.expanduser(path)
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith(('.tar.gz', '.tgz')):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith(('tar.bz2', '.tbz')):
        opener, mode = tarfile.open, 'r:bz2'
    elif path.endswith('.bz2'):
        opener, mode = bz2.BZ2File, 'rb'
        with open(path[:-4], 'wb') as fp_out, opener(path, 'rb') as fp_in:
            for data in iter(lambda: fp_in.read(100 * 1024), b''):
                fp_out.write(data)
        return
    else:
        raise (ValueError,
               "Could not extract `{}` as no extractor is found!".format(path))

    if to_directory is None:
        to_directory = os.path.abspath(os.path.join(path, os.path.pardir))
    cwd = os.getcwd()
    os.chdir(to_directory)

    try:
        file = opener(path, mode)
        try:
            file.extractall()
        finally:
            file.close()
    finally:
        os.chdir(cwd)
