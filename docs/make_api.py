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
            new_docstring_lst[idx-1] = '**%s**' % new_docstring_lst[idx-1]
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


def object_to_markdownpage(obj_name, obj, s=''):

    # header
    s += '## %s\n' % obj_name

    # function/class/method signature
    sig = str(inspect.signature(obj)).replace('(self, ', '(')
    s += '\n*%s%s*\n\n' % (obj_name, sig)

    # docstring body
    doc = str(inspect.getdoc(obj))
    ds = docstring_to_markdown(doc)
    s += '\n'.join(ds)

    # document methods
    if inspect.isclass(obj):
        s += '\n\n### Methods'
        methods = inspect.getmembers(obj)
        for m in methods:
            if not m[0].startswith('_'):
                sig = str(inspect.signature(m[1])).replace('(self, ', '(')
                s += '\n\n<hr>\n\n*%s%s*\n\n' % (m[0], sig)
                m_doc = docstring_to_markdown(str(inspect.getdoc(m[1])))
                s += '\n'.join(m_doc)

    return s + '\n\n'


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


def generate_api_docs(package, api_dir, clean=False, printlog=True):

    if printlog:
        print('\n\nGenerating Module Files\n%s\n' % (50 * '='))

    prefix = package.__name__ + "."

    # clear the previous version
    if clean:
        if os.path.isdir(api_dir):
            shutil.rmtree(api_dir)

    # get subpackages
    api_docs = {}
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
                if printlog:
                    print('created %s' % target_dir)

            # create markdown documents in memory
            for obj in classes + functions:
                md_path = os.path.join(target_dir, obj[0]) + '.md'
                if md_path not in api_docs:
                    api_docs[md_path] = object_to_markdownpage(obj_name=obj[0], obj=obj[1], s='')
                else:
                    api_docs[md_path] += object_to_markdownpage(obj_name=obj[0], obj=obj[1], s='')

    # write to files
    for d in sorted(api_docs):
        prev = ''
        if os.path.isfile(d):
            with open(d, 'r') as f:
                prev = f.read()
            if prev == api_docs[d]:
                msg = 'skipped'
            else:
                msg = 'updated'
        else:
            msg = 'created'

        if msg != 'skipped':
            with open(d, 'w') as f:
                f.write(api_docs[d])

        if printlog:
            print('%s %s' % (msg, d))


def summarize_methdods_and_functions(api_modules, out_dir, printlog=False, clean=True):

    if printlog:
        print('\n\nGenerating Subpackage Files\n%s\n' % (50 * '='))

    if clean:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        if printlog:
            print('created %s' % out_dir)

    subdir_paths = [os.path.join(api_modules, d)
                    for d in os.listdir(api_modules)
                    if not d.startswith('.')]

    out_files = [os.path.join(out_dir, os.path.basename(d)) + '.md'
                 for d in subdir_paths]

    for sub_p, out_f in zip(subdir_paths, out_files):
        module_paths = (os.path.join(sub_p, m)
                        for m in os.listdir(sub_p)
                        if not m.startswith('.'))

        new_output = []
        for p in module_paths:
            with open(p, 'r') as r:
                new_output.extend(r.readlines())

        msg = ''
        if not os.path.isfile(out_f):
            msg = 'created'

        if msg != 'created':
            with open(out_f, 'r') as f:
                prev = f.readlines()
            if prev != new_output:
                msg = 'updated'
            else:
                msg = 'skipped'

        if msg != 'skipped':
            with open(out_f, 'w') as f:
                f.write(''.join(new_output))

        if printlog:
            print('%s %s' % (msg, out_f))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
            description='Convert docstring into a markdown API documentation.',
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-n', '--package_name', default='mlxtend', help='Name of the package')
    parser.add_argument('-d', '--package_dir', default='../../mlxtend/', help="Path to the package's enclosing directory")
    parser.add_argument('-o1', '--output_module_api', default='../docs/sources/api_modules', help='Target directory for the module-level API Markdown files')
    parser.add_argument('-o2', '--output_subpackage_api', default='../docs/sources/api_subpackages', help='Target directory for the subpackage-level API Markdown files')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove previous API files')
    parser.add_argument('-s', '--silent', action='store_true', help='Suppress log printed to the screen')
    parser.add_argument('-v', '--version', action='version', version='v. 0.1')

    args = parser.parse_args()

    package = import_package(args.package_dir, args.package_name)
    generate_api_docs(package=package, api_dir=args.output_module_api, clean=args.clean, printlog=not(args.silent))
    summarize_methdods_and_functions(api_modules=args.output_module_api, out_dir=args.output_subpackage_api, printlog=not(args.silent), clean=args.clean)
