# API generator script
#
# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import string
import inspect
import os
import sys
import pkgutil
import shutil


def _obj_name(obj):
    if hasattr(obj, '__name__'):
        return obj.__name__


def docstring_to_markdown(docstring):
    new_docstring_lst = []

    for idx, line in enumerate(docstring.split('\n')):
        line = line.strip()
        if set(line) in ({'-'}, {'='}):
            new_docstring_lst[idx-1] = '**%s**\n' % new_docstring_lst[idx-1]
        elif line.startswith('>>>'):
            line = '    %s' % line
        new_docstring_lst.append(line)

    for idx, line in enumerate(new_docstring_lst[1:]):
        if line:
            if line.startswith('Description : '):
                new_docstring_lst[idx+1] = (new_docstring_lst[idx+1]
                                            .replace('Description : ', ''))
            elif ' : ' in line:
                line = line.replace(' : ', '` : ')
                new_docstring_lst[idx+1] = '\n- `%s\n' % line
            elif not line.startswith('*'):
                new_docstring_lst[idx+1] = '    %s' % line.lstrip()

    clean_lst = []
    for line in new_docstring_lst:
        if set(line.strip()) not in ({'-'}, {'='}):
            clean_lst.append(line)
    return clean_lst


def import_package(rel_path_to_package, package_name):
    try:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        curr_dir = os.path.dirname(os.path.realpath(os.getcwd()))
    package_path = os.path.join(curr_dir, rel_path_to_package)
    if package_path not in sys.path:
        sys.path = [package_path] + sys.path
    package = __import__(package_name)
    return package


def get_subpackages(package):
    "importer, subpackage_name"
    return [i for i in pkgutil.iter_modules(package.__path__) if i[2]]


def get_modules(package):
    "importer, subpackage_name"
    return [i for i in pkgutil.iter_modules(package.__path__)]


def get_functions_and_classes(package):
    classes, functions = [], []
    for name, member in inspect.getmembers(package):
        if not name.startswith('_'):
            if inspect.isclass(member):
                classes.append([name, member])
            elif inspect.isfunction(member):
                functions.append([name, member])
    return classes, functions


def generate_api_docs(package, api_dir, clean=True):
    prefix = package.__name__ + "."

    # clear the previous version
    if clean:
        if os.path.isdir(api_dir):
            shutil.rmtree(api_dir)

    # get subpackages
    for importer, pkg_name, is_pkg in pkgutil.iter_modules(
                                                           package.__path__,
                                                           prefix):
        if is_pkg:
            subpackage = __import__(pkg_name, fromlist="dummy")
            prefix = subpackage.__name__ + "."

            # get functions and classes
            classes, functions = get_functions_and_classes(subpackage)

            target_dir = os.path.join(api_dir, subpackage.__name__)

            # create the subdirs
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            # create markdown documents

            # class docs
            for cl in classes:
                with open(os.path.join(target_dir, cl[0]) + '.md', 'a') as f:
                    f.write('## %s\n\n' % cl[0])
                    sig = str(inspect.signature(cl[1])).replace('self, ', '')
                    f.write('\n\n*%s%s*\n\n' % (cl[0], sig))
                    class_doc = str(inspect.getdoc(cl[1]))
                    ds = docstring_to_markdown(class_doc)
                    formatted = '\n'.join(ds)
                    f.write(formatted)

                    # class method docs
                    f.write('\n\n### Methods\n\n')
                    members = inspect.getmembers(cl[1])
                    for m in members:
                        if not m[0].startswith('_'):
                            sig = str(inspect.signature(m[1])).replace('self, ', '')
                            f.write('\n\n*%s%s*\n\n' % (m[0], sig))
                            m_doc = docstring_to_markdown(str(inspect.getdoc(m[1])))
                            f.write('\n'.join(m_doc))

            # function docs
            for fn in functions:
                with open(os.path.join(target_dir, fn[0]) + '.md', 'a') as f:
                    f.write('## %s\n\n' % fn[0])
                    sig = str(inspect.signature(fn[1])).replace('self, ', '')
                    f.write('\n\n*%s%s*\n\n' % (fn[0], sig))
                    s = str(inspect.getdoc(fn[1]))
                    ds = docstring_to_markdown(s)
                    formatted = "\n".join(ds)
                    f.write(formatted)


def summarize_methdods_and_functions(api_topdir):
    subdir_paths = [os.path.join(api_topdir, d)
                    for d in os.listdir(api_topdir)
                    if not d.startswith('.')]

    for sp in subdir_paths:

        module_paths = [os.path.join(sp, m)
                        for m in os.listdir(sp)
                        if not m.startswith('.')]

        with open('%s.md' % sp, 'w') as f:
            for p in module_paths:
                with open(p, 'r') as r:
                    f.write(r.read() + '\n\n')

package = import_package('../../mlxtend/', 'mlxtend')
generate_api_docs(package=package, api_dir='../docs/sources/api', clean=True)
summarize_methdods_and_functions(api_topdir='../docs/sources/api')
