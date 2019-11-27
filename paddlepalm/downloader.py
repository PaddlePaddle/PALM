# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
try:
    from urllib.request import urlopen # Python 3
except ImportError:
    from urllib2 import urlopen # Python 2

# for https
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

_items = {
    'pretrain': {'ernie-en-uncased-large': 'https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz',
                 'bert-en-uncased-large': 'https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz',
                 'utils': None},
    'reader': {'utils': None},
    'backbone': {'utils': None},
    'tasktype': {'utils': None},
}

def _download(item, scope, path, silent=False):
    if not silent:
        print('Downloading {}: {} from {}...'.format(item, scope, _items[item][scope]))
    data_url = _items[item][scope]
    data_dir = path + '/' + item + '/' + scope
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir))
    filename = data_dir + '/' + data_url.split('/')[-1]

    def chunk_report(bytes_so_far, total_size):
        percent = float(bytes_so_far) / float(total_size)
        if percent > 1:
            percent = 1
        if not silent:
            print('\r>> Downloading... {:.1%}'.format(percent), end = "")

    def chunk_read(response, url, chunk_size = 16 * 1024, report_hook = None):
        total_size = response.info().getheader('Content-Length').strip()
        total_size = int(total_size)
        bytes_so_far = 0
        with open("%s" % filename, "wb") as f:
            while 1:
                chunk = response.read(chunk_size)
                f.write(chunk)
                f.flush() 
                bytes_so_far += len(chunk)
                if not chunk:
                    break
                if report_hook:
                    report_hook(bytes_so_far, total_size)
        return bytes_so_far

    response = urlopen(data_url)
    chunk_read(response, data_url, report_hook=chunk_report)
    
    if not silent:
        print(' done!')


def _convert():
    raise NotImplementedError()


def download(item, scope='all', path='.'):
    item = item.lower()
    scope = scope.lower()
    assert item in _items, '{} is not found. Support list: {}'.format(item, list(_items.keys()))
   
    if _items[item]['utils'] is not None:
        _download(item, 'utils', path, silent=True)

    if scope != 'all':
        assert scope in _items[item], '{} is not found. Support scopes: {}'.format(item, list(_items[item].keys()))
        _download(item, scope, path)
    else:
        for s in _items[item].keys():
            _download(item, s, path)


def ls(item=None, scope='all'):
    pass


