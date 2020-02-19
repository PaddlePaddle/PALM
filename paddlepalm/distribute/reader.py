
from . import gpu_dev_count, cpu_dev_count
try:
    import queue as Queue
except ImportError:
    import Queue
from threading import Thread

dev_count = gpu_dev_count if gpu_dev_count > 0 else cpu_dev_count

def yield_pieces(data, distribute_strategy, batch_size):
    """
    Args:
        distribute_strategy: support s=split, c=copy, u=unstack,
        """
    assert batch_size % dev_count == 0, "batch_size need to be integer times larger than dev_count."
    # print('data in yield pieces')
    # print(len(data))

    assert type(data) == type(distribute_strategy), [type(data), type(distribute_strategy)]
    assert len(data) == len(distribute_strategy), [len(data), len(distribute_strategy)]
    if isinstance(data, dict):
        keys = list(data.keys())
        data_list = [data[i] for i in keys]
        ds_list = [distribute_strategy[i] for i in keys]
    else:
        assert isinstance(data, list), "the input data must be a list or dict, and contained with multiple tensors."
        data_list = data
        ds_list = distribute_strategy
    stride = batch_size // dev_count
    p = stride
    # while p < len(data_list) + stride:
    while p <= batch_size:
        temp = []
        for d, s in zip(data_list, ds_list):
            s = s.strip().lower()
            if s == 's' or s == 'split':
                if p - stride >= len(d):
                    # print('WARNING: no more examples to feed empty devices')
                    temp = []
                    return
                temp.append(d[p-stride:p])
            elif s == 'u' or s == 'unstack':
                assert len(d) <= dev_count, 'Tensor size on dim 0 must be less equal to dev_count when unstack is applied.'
                if p//stride > len(d):
                    # print('WARNING: no more examples to feed empty devices')
                    return
                temp.append(d[p//stride-1])
            elif s == 'c' or s == 'copy':
                temp.append(d)
            else:
                raise NotImplementedError()
            
        p += stride
        if type(data) == dict:
            yield dict(zip(*[keys, temp]))
        else:
            # print('yielded pieces')
            # print(len(temp))
            yield temp


def data_feeder(reader, postprocess_fn=None, prefetch_steps=2, phase='train', is_multi=False):
    if postprocess_fn is None:
        def postprocess_fn(batch, id=-1, phase='train', is_multi=False):
            return batch

    def worker(reader, dev_count, queue):
        dev_batches = []
        for index, data in enumerate(reader()):
            if len(dev_batches) < dev_count:
                dev_batches.append(data)
            if len(dev_batches) == dev_count:
                queue.put((dev_batches, 0))
                dev_batches = []
        # For the prediction of the remained batches, pad more batches to 
        # the number of devices and the padded samples would be removed in
        # prediction outputs. 
        if len(dev_batches) > 0:
            num_pad = dev_count - len(dev_batches)
            for i in range(len(dev_batches), dev_count):
                dev_batches.append(dev_batches[-1])
            queue.put((dev_batches, num_pad))
        queue.put(None)

    queue = Queue.Queue(dev_count*prefetch_steps)
    p = Thread(
        target=worker, args=(reader, dev_count, queue))
    p.daemon = True
    p.start()
    while True:
        ret = queue.get()
        queue.task_done()
        if ret is not None:
            batches, num_pad = ret
            if dev_count > 1 and phase == 'train' and is_multi: 
                id = batches[0]['__task_id'][0]
            else:
                id = -1
            batch_buf = []
            flag_buf = []
            for idx, batch in enumerate(batches):
                # flag = num_pad == 0
                flag = idx-len(batches) < -num_pad
                # if num_pad > 0:
                #     num_pad -= 1
                batch = postprocess_fn(batch, id, phase, is_multi=is_multi)
                # batch = postprocess_fn(batch)
                batch_buf.append(batch)
                flag_buf.append(flag)
            yield batch_buf, flag_buf
        else:
            break
    queue.join()



def decode_fake(nums, mask, bs):
    bs //= dev_count
    n_t = 0
    for flag in mask:
        if not flag:
            break
        n_t = n_t + 1

    n_f = len(mask) - n_t
    p1 = nums - (n_t-1) * bs
    each_f = p1 / (n_f+1)
    return each_f * n_f

