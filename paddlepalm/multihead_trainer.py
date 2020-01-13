
from paddlepalm.distribute import gpu_dev_count, cpu_dev_count
from paddlepalm import Trainer

dev_count = 1 if gpu_dev_count <= 1 else gpu_dev_count
VERBOSE=False


class MultiHeadTrainer(Trainer):
    
    def __init__(self, trainers, reuse_flags=None):
        assert len(trainers) == len(mix_ratios)
        if reuse_flags is not None:
            assert len(reuse_flags) == len(trainers)

        self._trainers = trainers

    def build_forward(self, backbone, head_dict):

        num_heads = len(self._trainers)
        assert len(head_dict) == num_heads

        for t in trainers:
            assert t.name in head_dict
        
        train_prog = fluid.Program()
        train_init_prog = fluid.Program()

        def get_loss(i):
            head = head_dict[self._trainers[i].name]
            loss_var = self._trainers[i].build_forward(backbone, head, train_prog, train_init_prog)
            return loss_var
      
        task_fns = {}
        for i in range(num_heads):
            def task_loss():
                task_id = i
                return lambda: get_loss(task_id)
            task_fns[i] = task_loss()

        head_id_var = fluid.data(name="branch",shape=[1],dtype='int64')
        loss_var = layers.switch_case(
            branch_index=head_id_var,
            branch_fns=task_fns
        )
        self._head_id_var = head_id_var
        return loss_var

    def fit_readers(self, reader_dict, mix_ratio, ):
        
        num_heads = len(self._trainers)
        assert len(head_dict) == num_heads

        name_to_position = []
        joint_shape_and_dtypes = []
        iterators = []
        prefixes = []
        mrs = []
        net_inputs = []
        for t in trainers:
            assert t.name in reader_dict
            t.fit_reader(reader_dict[t.name])
            net_inputs.append(t._net_inputs)
            prefixes.append(t.name)
            mrs.append(t.mix_ratio)
            iterators.append(t._raw_iterator_fn())
            name_to_position.append(t._name_to_position)
            joint_shape_and_dtypes.append(t._shape_and_dtypes)

        iterator_fn = create_joint_iterator_fn(iterators, prefixes, joint_shape_and_dtypes, mrs, name_to_position, dev_count=dev_count, verbose=VERBOSE, return_type='dict')
        feed_batch_process_fn = reader_helper.create_multihead_feed_batch_process_fn(net_inputs)

        if gpu_dev_count > 1:
            distribute_feeder_fn = data_feeder(iterator_fn, feed_batch_process_fn)
        else:
            distribute_feeder_fn = iterator_fn

        if phase == 'train':
            self._train_reader = distribute_feeder_fn()
            self._feed_batch_process_fn = feed_batch_process_fn
        elif phase == 'predict':
            self._predict_reader = distribute_feeder_fn()
            self._pred_feed_batch_process_fn = feed_batch_process_fn
        
        
    def train(self):
        pass

    def train_one_step(self):
        pass

