
from __future__ import print_function
import os

_items = {
    'pretrain': {'ernie-en-uncased-large': 'http://xxxxx',
                 'xxx': 'xxx',
                 'utils': None}
    'reader': {'cls': 'xxx',
               'xxx': 'xxx',
               'utils': 'xxx'}
    'backbone': {xxx}
    'tasktype': {xxx}
}


def demo():
    raise NotImplementedError()


def _convert():
    raise NotImplementedError()

def download(item, scope='all', path='.'):
    item = item.lower()
    scope = scope.lower()
    assert item in items, '{} is not found. Support list: {}'.format(item, list(items.keys()))

    if not os.path.exists(path, item):
        os.makedirs(os.path.join(path, item))

    def _download(item, scope, silent=False):
        if not silent:
            print('downloading {}: {} from {}...'.format(item, scope, items[item][scope]), end='')
        urllib.downloadxxx(items[item][scope], path)
        if not silent:
            print('done!')
        
    if items['utils'] is not None:
        _download(item, 'utils', silent=True)

    if scope != 'all':
        assert scope in items[item], '{} is not found. Support scopes: {}'.format(item, list(items[item].keys()))
        _download(item, scope)
    else:
        for s in items[item].keys():
            _download(item, s)


def ls(item=None, scope='all'):
    pass


