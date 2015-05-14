import os
import subprocess
import ast
from mlxtend.file_io import find_files

rst_path = './sources/sphinx'
mkd_path = './sources/api'
mlxtend_path = '../mlxtend'

'''
# Generate reStructuredText
os.remove
out = subprocess.call(['sphinx-apidoc', '-o', rst_path, '/Users/sebastian/github/mlxtend/mlxtend', '-e', '-f', '--full', '--follow-links', '--maxdepth', '8'])

# Convert reStructuredText to markdown
files = [(os.path.join(rst_path, f), os.path.join(mkd_path, os.path.splitext(f)[0] + '.md')) for f in os.listdir(rst_path) if f.endswith('.rst')]
for f in files:
    out = subprocess.call(['pandoc', f[0], '--from=rst', '--to=markdown', '-o', f[1]])

# Get submodules

files = [f for f in os.listdir(mkd_path) if f.endswith('.md') and len(f.split('.'))>2]
for f in files:
    print("- ['api/%s', '%s']" % (f, os.path.splitext(f)[0]))
'''

py_files = find_files(substring='', path=mlxtend_path, check_ext='.py', recursive=True)
py_files = [p for p in py_files if not '__init__.py' in p]
print(py_files)

for f in py_files:
    with open(f, 'r') as fd:
         file_contents = fd.read()
    module = ast.parse(file_contents)
    function_definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]

    for fu in function_definitions:
        print(ast.get_docstring(fu))

    for cl in class_definitions:
        print(ast.get_docstring(cl))
        method_definitions = [node for node in cl if isinstance(node, ast.FunctionDef)]
        for m in method_definitions:
            print(ast.get_docstring(m))
    break
#files = [f for f in os.listdir(rst_path) if f.endswith('.rst')]
