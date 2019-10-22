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

BACKBONE_DIR='paddlepalm.backbone'
TASK_INSTANCE_DIR='paddlepalm.task_instance'
READER_DIR='paddlepalm.reader'
PARADIGM_DIR='paddlepalm.task_paradigm'
OPTIMIZER_DIR='paddlepalm.optimizer'
OPTIMIZE_METHOD='optimize'

REQUIRED_ARGS={
    'task_instance': str,
    'backbone': str,
    'optimizer': str,
    'learning_rate': float,
    'batch_size': int
    }

OPTIONAL_ARGS={
    'mix_ratio': str,
    'target_tag': str,
    'reuse_rag': str
    }

TASK_REQUIRED_ARGS={
    'paradigm': str,
    'reader': str,
    'train_file': str
    }

