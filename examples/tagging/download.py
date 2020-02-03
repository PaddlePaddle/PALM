#  -*- coding: utf-8 -*-

import os
import requests
import tarfile
import shutil
from tqdm import tqdm


def download(src, url):
    file_size = int(requests.head(url).headers['Content-Length'])

    header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
        '70.0.3538.67 Safari/537.36'
    }
    pbar = tqdm(total=file_size)
    resp = requests.get(url, headers=header, stream=True)

    with open(src, 'ab') as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)

    pbar.close()
    return file_size


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


