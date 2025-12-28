# API generator script
#
# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import os.path

import yaml

s = "# User Guide Index"

with open("mkdocs.yml", "r") as yml_cont:
    mkdocs_cfg = yaml.safe_load(yml_cont) or {}

nav = mkdocs_cfg.get("nav")
if not nav:
    raise KeyError("'nav' section missing from mkdocs.yml")

user_guide_entry = next(
    (item for item in nav if isinstance(item, dict) and "User Guide" in item),
    None,
)
if not user_guide_entry:
    raise KeyError("'User Guide' section missing from mkdocs.yml")

usr_gd = user_guide_entry["User Guide"]
for dct in usr_gd[1:]:
    subpk = list(dct.keys())[0]
    s += "\n\n## `%s`" % subpk
    for obj in dct[subpk]:
        bsname = os.path.basename(obj).split(".md")[0]
        s += "\n- [%s](%s)" % (bsname, obj)

usr_gd_file = os.path.join("sources", "USER_GUIDE_INDEX.md")

with open(usr_gd_file, "w") as f:
    f.write(s)
