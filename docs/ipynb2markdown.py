# IPython Notebook to Markdown conversion script
#
# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import subprocess
import glob
import shutil
import os
import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension
from nbconvert.exporters import MarkdownExporter


class ImgExtractor(Treeprocessor):
    def run(self, doc):
        self.markdown.images = []
        for image in doc.findall('.//img'):
            self.markdown.images.append(image.get('src'))


class ImgExtExtension(Extension):
    def extendMarkdown(self, md, md_globals):
        img_ext = ImgExtractor(md)
        md.treeprocessors.add('imgext', img_ext, '>inline')


def ipynb_to_md(ipynb_path):
    orig_path = os.getcwd()
    os.chdir(os.path.dirname(ipynb_path))
    file_name = os.path.basename(ipynb_path)
    subprocess.call(['python', '-m', 'nbconvert',
                     '--to', 'markdown', file_name])

    new_s = []
    md_name = file_name.replace('.ipynb', '.md')
    with open(md_name, 'r') as f:
        for line in f:
            if line.startswith('#'):
                new_s.append(line)
                break
        for line in f:
            if line.startswith(('## API', '# API')):
                new_s.append(line)
                new_s.append('\n')
                break
            new_s.append(line)
        for line in f:
            if line.lstrip().startswith('#'):
                break
        for line in f:
            if line.lstrip().startswith('```'):
                continue
            else:
                new_s.append(line[4:])

    with open(md_name, 'w') as f:
        f.write(''.join(new_s))
    os.chdir(orig_path)


# md = markdown.Markdown(extensions=[ImgExtExtension()])
# html = md.convert(data)
# print(md.images)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
            description='Convert docstring into a markdown API documentation.',
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--ipynb',
                        help='Path to the IPython file')

    parser.add_argument('-a', '--all',
                        help='Path to parse all ipynb recursively')

    parser.add_argument('-v', '--version',
                        action='version',
                        version='v. 0.1')

    args = parser.parse_args()

    if args.all and args.ipynb:
        raise AttributeError('Conflicting flags --ipynb and --all; choose one')

    if args.ipynb:
        ipynb_to_md(ipynb_path=args.ipynb)
    else:
        tree = os.walk(args.all)
        for d in tree:
            filenames = glob.glob(os.path.join(d[0], '*'))
            for f in filenames:
                if f.endswith('.ipynb'):
                    print(f)

                    ipynb_to_md(ipynb_path=f)
