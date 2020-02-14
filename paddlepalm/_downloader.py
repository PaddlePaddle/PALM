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
import requests
import tarfile
import shutil
try:
    from urllib.request import urlopen # Python 3
except ImportError:
    from urllib2 import urlopen # Python 2
from collections import OrderedDict
import ssl

__all__ = ["download", "ls"]

# for https
ssl._create_default_https_context = ssl._create_unverified_context



_pretrain = (('RoBERTa-zh-base', 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz'),
            ('RoBERTa-zh-large', 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz'),
            ('ERNIE-v2-en-base', 'https://ernie.bj.bcebos.com/ERNIE_Base_en_stable-2.0.0.tar.gz'),
            ('ERNIE-v2-en-large', 'https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz'),
            ('XLNet-cased-base','https://xlnet.bj.bcebos.com/xlnet_cased_L-12_H-768_A-12.tgz'),
            ('XLNet-cased-large','https://xlnet.bj.bcebos.com/xlnet_cased_L-24_H-1024_A-16.tgz'),
            ('ERNIE-v1-zh-base','https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz'),
            ('ERNIE-v1-zh-base-max-len-512','https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz'),
            ('BERT-en-uncased-large-whole-word-masking','https://bert-models.bj.bcebos.com/wwm_uncased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-cased-large-whole-word-masking','https://bert-models.bj.bcebos.com/wwm_cased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-uncased-base', 'https://bert-models.bj.bcebos.com/uncased_L-12_H-768_A-12.tar.gz'),
            ('BERT-en-uncased-large', 'https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-cased-base','https://bert-models.bj.bcebos.com/cased_L-12_H-768_A-12.tar.gz'),
            ('BERT-en-cased-large','https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-multilingual-uncased-base','https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz'),
            ('BERT-multilingual-cased-base','https://bert-models.bj.bcebos.com/multi_cased_L-12_H-768_A-12.tar.gz'),
            ('BERT-zh-base','https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz'),
            ('utils', None))
_vocab = (('utils', None),('utils', None))
_backbone =(('utils', None),('utils', None))
_head = (('utils', None),('utils', None))
_reader = (('utils', None),('utils', None))

_items = (('pretrain', OrderedDict(_pretrain)),
        ('vocab', OrderedDict(_vocab)), 
        ('backbone', OrderedDict(_backbone)),
        ('head', OrderedDict(_head)),
        ('reader', OrderedDict(_reader))
)
_items = OrderedDict(_items)

def _download(item, scope, path, silent=False, convert=False):
    data_url = _items[item][scope]
    if data_url == None:
        return
    if not silent:
        print('Downloading {}: {} from {}...'.format(item, scope, data_url))
    data_dir = path + '/' + item + '/' + scope
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir))
    data_name = data_url.split('/')[-1]
    filename = data_dir + '/' + data_name

    # print process
    def _chunk_report(bytes_so_far, total_size):
        percent = float(bytes_so_far) / float(total_size)
        if percent > 1:
            percent = 1
        if not silent:
            print('\r>> Downloading... {:.1%}'.format(percent), end = "")
    
    # copy to local
    def _chunk_read(response, url, chunk_size = 16 * 1024, report_hook = None):
        total_size = int(requests.head(url).headers['Content-Length'])
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
    _chunk_read(response, data_url, report_hook=_chunk_report)
    
    if not silent:
        print(' done!')
    
    if item == 'pretrain':
        if not silent:
            print ('Extracting {}...'.format(data_name), end=" ")
        if os.path.exists(filename):
            tar = tarfile.open(filename, 'r')
            tar.extractall(path = data_dir)
            tar.close()
            os.remove(filename)
        if len(os.listdir(data_dir))==1:
            source_path = data_dir + '/' + data_name.split('.')[0]
            fileList = os.listdir(source_path)
            for file in fileList:
                filePath = os.path.join(source_path, file)
                shutil.move(filePath, data_dir)
            os.removedirs(source_path)
        if not silent:
            print ('done!')
        if convert:
            if not silent:
                print ('Converting params...', end=" ")
            _convert(data_dir, silent)
        if not silent:
            print ('done!')


def _convert(path, silent=False):
    if os.path.isfile(path + '/params/__palminfo__'):
        if not silent:
            print ('already converted.')
    else:
        if os.path.exists(path + '/params/'):
            os.rename(path + '/params/', path + '/params1/')
            os.mkdir(path + '/params/')
            tar_model = tarfile.open(path + '/params/' + '__palmmodel__', 'w')
            tar_info = open(path + '/params/'+ '__palminfo__', 'w')
            for root, dirs, files in os.walk(path + '/params1/'):
                for file in files:
                    src_file = os.path.join(root, file)
                    tar_model.add(src_file, '__paddlepalm_' + file)
                    tar_info.write('__paddlepalm_' + file)
                    os.remove(src_file)
            tar_model.close()
            tar_info.close()
            os.removedirs(path + '/params1/') 

def download(item, scope='all', path='.'):
    """download an item. The available scopes and contained items can be showed with `paddlepalm.downloader.ls`.

    Args:
        item: the item to download.
        scope: the scope of the item to download.
        path: the target dir to download to. Default is `.`, means current dir.
    """
    # item = item.lower()
    # scope = scope.lower()
    assert item in _items, '{} is not found. Support list: {}'.format(item, list(_items.keys()))
   
    if _items[item]['utils'] is not None:
        _download(item, 'utils', path, silent=True)

    if scope != 'all':
        assert scope in _items[item], '{} is not found. Support scopes: {}'.format(scope, list(_items[item].keys()))
        _download(item, scope, path)
    else:
        for s in _items[item].keys():
            _download(item, s, path)


def _ls(item, scope, l = 10):
    if scope != 'all':
        assert scope in _items[item], '{} is not found. Support scopes: {}'.format(scope, list(_items[item].keys()))
        print ('{}'.format(scope))
    else:
        for s in _items[item].keys():
            if s == 'utils':
                continue
            print ('  => '+s)

def ls(item='all', scope='all'):
    
    if scope == 'utils':
        return
    if item != 'all':
        assert item in _items, '{} is not found. Support scopes: {}'.format(item, list(_items.keys()))
        print ('Available {} items:'.format(item))
        _ls(item, scope)
    else:
        l = max(map(len, _items.keys()))
        for i in _items.keys():
            print ('Available {} items: '.format(i))
            _ls(i, scope, l)


    
