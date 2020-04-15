#  -*- coding: utf-8 -*-
from __future__ import print_function
import os
import tarfile
import shutil
import sys
import urllib
URLLIB=urllib
if sys.version_info >= (3, 0):
    import urllib.request
    URLLIB=urllib.request

def download(src, url):
    def _reporthook(count, chunk_size, total_size):
        bytes_so_far = count * chunk_size
        percent = float(bytes_so_far) / float(total_size)
        if percent > 1:
            percent = 1
        print('\r>> Downloading... {:.1%}'.format(percent), end="")

    URLLIB.urlretrieve(url, src, reporthook=_reporthook)

abs_path = os.path.abspath(__file__)
download_url = "https://baidu-nlp.bj.bcebos.com/dmtk_data_1.0.0.tar.gz"
downlaod_path = os.path.join(os.path.dirname(abs_path), "dmtk_data_1.0.0.tar.gz")
target_dir = os.path.dirname(abs_path)
download(downlaod_path, download_url)

tar = tarfile.open(downlaod_path)
tar.extractall(target_dir)
os.remove(downlaod_path)

shutil.rmtree(os.path.join(target_dir, 'data/dstc2/'))
shutil.rmtree(os.path.join(target_dir, 'data/mrda/'))
shutil.rmtree(os.path.join(target_dir, 'data/multi-woz/'))
shutil.rmtree(os.path.join(target_dir, 'data/swda/'))
shutil.rmtree(os.path.join(target_dir, 'data/udc/'))
print(" done!")
