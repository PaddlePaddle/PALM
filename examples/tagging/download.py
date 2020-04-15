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
download_url = "https://ernie.bj.bcebos.com/task_data_zh.tgz"
downlaod_path = os.path.join(os.path.dirname(abs_path), "task_data_zh.tgz")
target_dir = os.path.dirname(abs_path)
download(downlaod_path, download_url)

tar = tarfile.open(downlaod_path)
tar.extractall(target_dir)
os.remove(downlaod_path)

abs_path = os.path.abspath(__file__)
dst_dir = os.path.join(os.path.dirname(abs_path), "data")
if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
    os.makedirs(dst_dir)

for file in os.listdir(os.path.join(target_dir, 'task_data', 'msra_ner')):
    shutil.move(os.path.join(target_dir, 'task_data', 'msra_ner', file), dst_dir)

shutil.rmtree(os.path.join(target_dir, 'task_data'))
print(" done!")
