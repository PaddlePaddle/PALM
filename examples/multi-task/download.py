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

