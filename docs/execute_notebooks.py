import os
import glob
import subprocess

tree = os.walk('sources')
for d in tree:
    filenames = glob.glob(os.path.join(d[0], '*'))
    for f in filenames:
        print(f)
        if f.endswith('.ipynb'):
            subprocess.call(['jupyter',
                             'nbconvert',
                             '--to',
                             'notebook',
                             '--execute',
                             f])
