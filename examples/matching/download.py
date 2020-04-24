#  -*- coding: utf-8 -*-
from __future__ import print_function
import os
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
data_dir = os.path.join(os.path.dirname(abs_path), "data")
if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
    os.makedirs(data_dir)

download_url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
downlaod_path = os.path.join(data_dir, "quora_duplicate_questions.tsv")
download(downlaod_path, download_url)
print(" done!")
