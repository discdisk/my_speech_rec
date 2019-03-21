import chainer

# from __future__ import division

# import numpy

# from chainer.dataset import iterator
# from chainer.iterators.order_samplers import ShuffleOrderSampler


        



class My_SerialIterator(chainer.iterators.SerialIterator):
    def __init__(self, dataset, batch_size,
                 repeat=True, shuffle=None, order_sampler=None):
        super(My_SerialIterator, self).__init__(dataset, batch_size,
                 repeat=True, shuffle=None, order_sampler=None)
        self.len_limit = 15



    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self._epoch_size

        if self._order is None:
            ##modify
            batch = []
            count=0
            while len(batch)<self.batch_size:
                if count+i<N:
                    if 3<len(self.dataset[i+count][1][0])<self.len_limit:
                        batch.append(self.dataset[i+count])
                        i_end = i + count
                else: 
                    i_end=N+self.batch_size-len(batch)
                    break
                count+=1

        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    new_order = self.order_sampler(self._order, i)
                    if len(self._order) != len(new_order):
                        raise ValueError('The size of order does not match '
                                         'the size of the previous order.')
                    self._order = new_order
                if rest > 0:
                    if self._order is None:
                        batch.extend(self.dataset[:rest])
                    else:
                        batch.extend([self.dataset[index]
                                      for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
            self.len_limit += 3
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

