# API generator script
#
# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import yaml
import os.path

s = "# User Guide Index"

yml_cont = open('mkdocs.yml', 'r')
usr_gd = yaml.load(yml_cont)['pages'][1]['User Guide']
for dct in usr_gd[1:]:
    subpk = list(dct.keys())[0]
    s += '\n\n## `%s`' % subpk
    for obj in dct[subpk]:
        bsname = os.path.basename(obj).split('.md')[0]
        s += '\n- [%s](%s)' % (bsname, obj)

usr_gd_file = os.path.join('sources', 'USER_GUIDE_INDEX.md')

with open(usr_gd_file, 'w') as f:
    f.write(s)
