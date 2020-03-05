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
from paddlepalm.reader.utils.reader4ernie import ClassifyReader as CLSReader


class MatchReader(Reader):
    """
    The reader completes the loading and processing of matching-like task (e.g, query-query, question-answer, text similarity, natural language inference) dataset. Supported file format: tsv. 
    
    For pointwise learning strategy, there should be two fields in training dataset file, i.e., `text_a`, `text_b` and `label`. For pairwise learning, there should exist three fields, i.e., `text_a`, `text_b` and `text_b_neg`. For predicting, only `text_a` and `text_b` are required.
    
    A pointwise learning case shows as follows:
    ```
    label [TAB] text_a [TAB] text_b
    1 [TAB] Today is a good day. [TAB] what a nice day!
    0 [TAB] Such a terriable day! [TAB] There is a dog.
    1 [TAB] I feel lucky to meet you, dear. [TAB] You are my lucky, darling.
    1 [TAB] He likes sunshine and I like him :). [TAB] I like him. He like sunshine.
    0 [TAB] JUST! GO! OUT! [TAB] Come in please.
    ```
    A pairwise learning case shows as follows:
    text_a [TAB] text_b [TAB] text_b_neg
    Today is a good day. [TAB] what a nice day! [TAB] terriable day!
    Such a terriable day! [TAB] So terriable today! [TAB] There is a dog.
    I feel lucky to meet you, dear. [TAB] You are my lucky, darling. [TAB] Buy some bananas, okey?
    He likes sunshine and I like him :). [TAB] I like him. He like sunshine. [TAB] He has a dog.
    JUST! GO! OUT! [TAB] go out now! [TAB] Come in please.

    CAUTIOUS: the HEADER is required for each dataset file! And fields (columns) should be splited by Tab (\\t).

    """
    
    def __init__(self, vocab_path, max_len, tokenizer='wordpiece', lang='en', seed=None, \
        do_lower_case=False, learning_strategy='pointwise', phase='train', dev_count=1, print_prefix=''): 
        """Create a new Reader for classification task data.

        Args:
          vocab_path: the vocab file path to do tokenization and token_ids generation.
          max_len: The maximum length of the sequence (after word segmentation). The part exceeding max_len will be removed from right.
          tokenizer: string type. The name of the used tokenizer. A tokenizer is to convert raw text into tokens. Avaliable tokenizers: wordpiece.
          lang: the language of dataset. Supported language: en (English), cn (Chinese). Default is en (English). 
          seed: int type. The random seed to shuffle dataset. Default is None, means no use of random seed.
          do_lower_case: bool type. Whether to do lowercase on English text. Default is False. This argument only works on English text.
          learning_strategy: string type. This only works for training phase. Available strategies: pointwise, pairwise.
          phase: the running phase of this reader. Supported phase: train, predict. Default is train.

        Return:
            a Reader object for matching-like task.
        """

        Reader.__init__(self, phase)

        assert lang.lower() in ['en', 'cn', 'english', 'chinese'], "supported language: en (English), cn (Chinese)."
        assert phase in ['train', 'predict'], "supported phase: train, predict."

        for_cn = lang.lower() == 'cn' or lang.lower() == 'chinese'

        self._register.add('token_ids')
        if phase == 'train':
            if learning_strategy == 'pointwise':
                self._register.add('label_ids')
            if learning_strategy == 'pairwise':
                self._register.add('token_ids_neg')
                self._register.add('position_ids_neg')
                self._register.add('segment_ids_neg')
                self._register.add('input_mask_neg')
                self._register.add('task_ids_neg')

        self._is_training = phase == 'train'
        self._learning_strategy = learning_strategy


        match_reader = CLSReader(vocab_path,
                                max_seq_len=max_len,
                                do_lower_case=do_lower_case,
                                for_cn=for_cn,
                                random_seed=seed,
                                learning_strategy = learning_strategy)
            
        self._reader = match_reader
        self._dev_count = dev_count
        self._phase = phase


    @property
    def outputs_attr(self):
        attrs = {"token_ids": [[-1, -1], 'int64'],
                "position_ids": [[-1, -1], 'int64'],
                "segment_ids": [[-1, -1], 'int64'],
                "input_mask": [[-1, -1, 1], 'float32'],
                "task_ids": [[-1, -1], 'int64'],
                "label_ids": [[-1], 'int64'],
                "token_ids_neg": [[-1, -1], 'int64'],
                "position_ids_neg": [[-1, -1], 'int64'],
                "segment_ids_neg": [[-1, -1], 'int64'],
                "input_mask_neg": [[-1, -1, 1], 'float32'],
                "task_ids_neg": [[-1, -1], 'int64']
                }
        return self._get_registed_attrs(attrs)


    def load_data(self, input_file, batch_size, num_epochs=None, \
                  file_format='tsv', shuffle_train=True):
        """Load matching data into reader. 

        Args:
            input_file: the dataset file path. File format should keep consistent with `file_format` argument.
            batch_size: number of examples for once yield. CAUSIOUS! If your environment exists multiple GPU devices (marked as dev_count), the batch_size should be divided by dev_count with no remainder!
            num_epochs: the travelsal times of input examples. Default is None, means once for single-task learning and automatically calculated for multi-task learning. This argument only works on train phase.
            file_format: the file format of input file. Supported format: tsv. Default is tsv.
            shuffle_train: whether to shuffle training dataset. Default is True. This argument only works on training phase.

        """
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._data_generator = self._reader.data_generator( \
            input_file, batch_size, num_epochs if self._phase == 'train' else 1, \
            shuffle=shuffle_train if self._phase == 'train' else False, \
            phase=self._phase)

    def _iterator(self): 

        
        names = ['token_ids', 'segment_ids', 'position_ids', 'task_ids', 'input_mask', 'label_ids', \
            'token_ids_neg', 'segment_ids_neg', 'position_ids_neg', 'task_ids_neg', 'input_mask_neg']
        
        if self._learning_strategy == 'pairwise':
            names.remove('label_ids')


        for batch in self._data_generator():
            outputs = {n: i for n,i in zip(names, batch)}
            ret = {}
            # TODO: move runtime shape check here
            for attr in self.outputs_attr.keys():
                ret[attr] = outputs[attr]
            yield ret

    @property
    def num_examples(self):
        return self._reader.get_num_examples(phase=self._phase)

    @property
    def num_epochs(self):
        return self._num_epochs

