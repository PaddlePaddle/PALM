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

import paddlepalm as palm
import sys
import argparse
 
# create parser
parser = argparse.ArgumentParser(prog='download_models.py', usage='%(prog)s < -l | -d <model_name> > [-h] ',description = 'Download pretrain models for initializing params of backbones. ')
parser1= parser.add_argument_group("pretrain models")
parser1.add_argument('-l','--list', action = 'store_true', help = 'show the list of pretrain models', default = False)
parser1.add_argument('-d','--download', action = 'store', help = 'download pretrain models') 
args = parser.parse_args()

if(args.list):
  palm.downloader.ls('pretrain')
elif(args.download):
  palm.downloader.download('pretrain', args.download)
else:
  print (parser.parse_args(['-h']))
