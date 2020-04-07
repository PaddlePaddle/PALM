# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2019  Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
################################################################################
"""
Setup script.
Authors: zhouxiangyang(zhouxiangyang@baidu.com)
Date:    2020/2/4 00:00:01
"""
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="paddlepalm",
    version="2.0.1",
    author="PaddlePaddle",
    author_email="zhangyiming04@baidu.com",
    description="a flexible, general and easy-to-use NLP large-scale pretraining and multi-task learning framework.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/PALM",
    # packages=setuptools.find_packages(),
    packages = ['paddlepalm', 
        'paddlepalm.backbone', 
        'paddlepalm.backbone.utils', 
        'paddlepalm.optimizer',
        'paddlepalm.reader', 
        'paddlepalm.reader.utils', 
        'paddlepalm.head', 
        'paddlepalm.distribute', 
        'paddlepalm.lr_sched', 
        'paddlepalm.tokenizer', 
        'paddlepalm.utils'],
    package_dir={'paddlepalm':'./paddlepalm',
                 'paddlepalm.backbone':'./paddlepalm/backbone',
                 'paddlepalm.backbone.utils':'./paddlepalm/backbone/utils',
                 'paddlepalm.optimizer':'./paddlepalm/optimizer',
                 'paddlepalm.lr_sched': './paddlepalm/lr_sched',
                 'paddlepalm.distribute': './paddlepalm/distribute',
                 'paddlepalm.reader':'./paddlepalm/reader',
                 'paddlepalm.reader.utils':'./paddlepalm/reader/utils',
                 'paddlepalm.head':'./paddlepalm/head',
                 'paddlepalm.tokenizer':'./paddlepalm/tokenizer',
                 'paddlepalm.utils':'./paddlepalm/utils'},
    platforms = "any",
    license='Apache 2.0',
    classifiers = [
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
          ],
    install_requires = [
        'paddlepaddle-gpu>=1.7.0'
    ]
)


