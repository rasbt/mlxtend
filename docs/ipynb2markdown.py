# IPython Notebook to Markdown conversion script
#
# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import subprocess
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
            if line.startswith('# API'):
                new_s.append(line)
                new_s.append('\n')
                break
            new_s.append(line)
        for line in f:
            if line.lstrip().startswith('#'):
                break
        for line in f:
            new_s.append(line[4:])

    with open(md_name, 'w') as f:
        f.write(''.join(new_s))


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

    parser.add_argument('-v', '--version',
                        action='version',
                        version='v. 0.1')

    args = parser.parse_args()

    ipynb_to_md(ipynb_path=args.ipynb)
