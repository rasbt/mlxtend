# content of conftest.py
import os

collect_ignore = []
if 'TRAVIS' in os.environ:
    collect_ignore.append("tests/test_extract_face_landmarks.py")
    collect_ignore.append("tests/test_eyepad_align.py")
