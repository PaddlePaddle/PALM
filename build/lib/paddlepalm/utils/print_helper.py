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

MAXLEN = 70
def print_dict(dic, title=""):

    if title:
        title = ' ' + title + ' '
        left_len = (MAXLEN - len(title)) // 2
        title = '-' * left_len + title
        right_len = MAXLEN - len(title)
        title = title + '-' * right_len
    else:
        title = '-' * MAXLEN
    print(title)
    for name in dic:
        print("{: <25}\t{}".format(str(name), str(dic[name])))
    print("")
    # print("-" * MAXLEN + '\n')
