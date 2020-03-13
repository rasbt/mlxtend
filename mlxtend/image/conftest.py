# content of conftest.py
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


collect_ignore = []
if 'TRAVIS' in os.environ or os.environ.get('TRAVIS') == 'true':
    collect_ignore.append("tests/test_extract_face_landmarks.py")
    collect_ignore.append("tests/test_eyepad_align.py")
