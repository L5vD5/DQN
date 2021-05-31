from collections import deque
import random
from collections import namedtuple
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, buffer_size, odim, adim, minibatch_size=32):
        self.buffer_size = buffer_size
        self.o_buf = np.zeros((buffer_size, odim), dtype=np.float32)
        self.a_buf = np.zeros((buffer_size, adim), dtype=np.float32)
        self.r_buf = np.zeros(buffer_size, dtype=np.float32)
        self.o1_buf = np.zeros((buffer_size, odim), dtype=np.float32)
        self.d_buf = np.zeros(buffer_size, dtype=np.float32)
        self.minibatch_size = minibatch_size
        self._full = False

        self.ptr = 0

    def is_full(self):
        return self._full

    @property
    def cur_ptr(self):
        return self.ptr

    def get_minibatch_indices(self):
        indices = []
        while len(indices) < self.minibatch_size:
            while True:
                if self.is_full():
                    index = np.random.randint(low=0, high=self.buffer_size, dtype=np.int32)
                else:
                    index = np.random.randint(low=0, high=self.cur_ptr, dtype=np.int32)

                if np.any([sample.terminal for sample in self._memory[index - self.history_len:index]]):
                    continue
                indices.append(index)
                break
        return indices

    def generate_minibatch_samples(self, indices):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = [], [], [], [], []

        for index in indices:
            selected_mem = self._memory[index]
            state_batch.append(tf.constant(selected_mem.state, tf.float32))
            action_batch.append(tf.constant(selected_mem.action, tf.int32))
            reward_batch.append(tf.constant(selected_mem.reward, tf.float32))
            next_state_batch.append(tf.constant(selected_mem.next_state, tf.float32))
            terminal_batch.append(tf.constant(selected_mem.terminal, tf.bool))

        return tf.stack(state_batch, axis=0), tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), tf.stack(
            next_state_batch, axis=0), tf.stack(terminal_batch, axis=0)

    def append(self, o, a, r, o1, d):
        # o, a, r, o1, d = [tf.convert_to_tensor(p) for p in [o, a, r, o1, d]]
        assert self.ptr < self.buffer_size  # buffer has to have room so you can store
        self.o_buf[self.ptr] = o
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.o1_buf[self.ptr] = o1
        self.d_buf[self.ptr] = d
        self.ptr += 1