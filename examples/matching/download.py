#  -*- coding: utf-8 -*-

import os
import requests
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
data_dir = os.path.join(os.path.dirname(abs_path), "data")
if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
    os.makedirs(data_dir)

download_url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
downlaod_path = os.path.join(data_dir, "quora_duplicate_questions.tsv")
download(downlaod_path, download_url)
