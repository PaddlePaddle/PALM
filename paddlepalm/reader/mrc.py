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

from paddlepalm.reader.base_reader import Reader
from paddlepalm.reader.utils.reader4ernie import MRCReader
import numpy as np

class MrcReader(Reader):

    def __init__(self, vocab_path, max_len, max_query_len, doc_stride, tokenizer='FullTokenizer', lang='en', seed=None, do_lower_case=False, \
        remove_noanswer=True, phase='train', dev_count=1, print_prefix=''):
        """
        Args:
            phase: train, eval, pred
            lang: en, ch, ...
        """

        Reader.__init__(self, phase)


        assert lang.lower() in ['en', 'cn', 'english', 'chinese'], "supported language: en (English), cn (Chinese)."
        assert phase in ['train', 'predict'], "supported phase: train, predict."

        for_cn = lang.lower() == 'cn' or lang.lower() == 'chinese'


        self._register.add('token_ids')
        if phase == 'train':
            self._register.add("start_positions")
            self._register.add("end_positions")
        else:
            self._register.add("unique_ids")
            

        self._is_training = phase == 'train'

        mrc_reader = MRCReader(vocab_path,
                                max_seq_len=max_len,
                                do_lower_case=do_lower_case,
                                tokenizer=tokenizer,
                                doc_stride=doc_stride,
                                remove_noanswer=remove_noanswer,
                                max_query_length=max_query_len,
                                for_cn=for_cn,
                                random_seed=seed)
        self._reader = mrc_reader

        self._phase = phase
        self._dev_count = dev_count
 

    @property
    def outputs_attr(self):
        attrs = {"token_ids": [[-1, -1], 'int64'],
                "position_ids": [[-1, -1], 'int64'],
                "segment_ids": [[-1, -1], 'int64'],
                "input_mask": [[-1, -1, 1], 'float32'],
                "start_positions": [[-1], 'int64'],
                "end_positions": [[-1], 'int64'],
                "task_ids": [[-1, -1], 'int64'],
                "unique_ids": [[-1], 'int64']
                }
        return self._get_registed_attrs(attrs)

    @property
    def epoch_outputs_attr(self):
        if not self._is_training:
            return {"examples": None,
                    "features": None}

    def load_data(self, input_file, batch_size, num_epochs=None, file_format='csv', shuffle_train=True):
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._data_generator = self._reader.data_generator( \
            input_file, batch_size, num_epochs if self._phase == 'train' else 1, \
            shuffle=shuffle_train if self._phase == 'train' else False, \
            phase=self._phase)
    def _iterator(self): 

        names = ['token_ids', 'segment_ids', 'position_ids', 'task_ids', 'input_mask', 
            'start_positions', 'end_positions', 'unique_ids']
        
        if self._is_training:
            names.remove('unique_ids')
        
        for batch in self._data_generator():
            outputs = {n: i for n,i in zip(names, batch)}
            ret = {}
            # TODO: move runtime shape check here
            for attr in self.outputs_attr.keys():
                ret[attr] = outputs[attr]
            if not self._is_training:
                assert 'unique_ids' in ret, ret
            yield ret
    

    def get_epoch_outputs(self):

        return {'examples': self._reader.get_examples(self._phase),
                'features': self._reader.get_features(self._phase)}

    @property
    def num_examples(self):
        return self._reader.get_num_examples(phase=self._phase)

    @property
    def num_epochs(self):
        return self._num_epochs

