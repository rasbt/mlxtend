# API generator script
#
# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import yaml
import os
import subprocess
from mlxtend import __version__

yaml_path = "./mkdocs.yml"
source_path = os.path.join(os.path.dirname(yaml_path), 'sources')
md_out_path = "./temp.md"

with open(yaml_path, 'r') as f:
    content = f.read()

tree = yaml.load(content)

mkdocs = []


def get_leaf_nodes(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            get_leaf_nodes(v)
    elif isinstance(tree, list) or isinstance(tree, tuple):
        for ele in tree:
            get_leaf_nodes(ele)
    else:
        mkdocs.append(tree)

get_leaf_nodes(tree['pages'])
mkdocs = [s for s in mkdocs if 'api_subpackages' not in s and
          'USER_GUIDE_INDEX.md' not in s]


def abs_imagepath(line, md_path):
    elements = line.split('](')
    rel_path = elements[1].strip().rstrip(')')
    img_path = os.path.join(md_path, rel_path)
    img_link = '%s](%s)\n' % (elements[0], img_path)
    img_link = img_link.replace('/./', '/')
    return img_link


def gen_title(fname):
    stem, title = os.path.split(fname)
    title = title.rstrip('.md')
    s = '# `%s.%s`' % (os.path.split(stem)[1], title)
    return s


with open(md_out_path, 'w') as f_out:
    meta = r"""---
title: Mlxtend %s
subtitle: Library Documentation
author: Sebastian Raschka
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[LO,LE]{\thepage}
    - \fancyfoot[CE,CO]{}
---

""" % __version__
    f_out.write(meta)
    for md in mkdocs:
        md_path = os.path.join(source_path, md)
        img_path = os.path.dirname(md_path)

        with open(md_path, 'r') as f_in:
            content = f_in.readlines()
            if md.startswith('user_guide'):

                title = gen_title(md)
                f_out.write(title + '\n')
                if content[0].startswith('# '):
                    content = content[1:]

            for line in content:
                if '![png]' in line:
                    line = line.replace('![png]', '![]')
                if '.svg' in line:
                    continue
                if line.startswith('!['):
                    line = abs_imagepath(line, img_path)
                f_out.write(line)
            f_out.write('\n\n')

subprocess.check_call(['pandoc', '-N', 'temp.md', '--output=mlxtend.pdf',
                       '--toc',
                       '--normalize', '--smart', '--latex-engine=xelatex',
                       '--toc-depth=4', '--highlight-style=pygments',
                       '--template=pdftemplate.tex'])

os.remove(md_out_path)
