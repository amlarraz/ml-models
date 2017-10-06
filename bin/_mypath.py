import os, sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

thisdir = os.path.dirname(__file__)
#Add lib path to the system path
libdir = os.path.join(thisdir, '../lib')
add_path(libdir)