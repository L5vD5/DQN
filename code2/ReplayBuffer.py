from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def sample(self, batch_size):
        size=batch_size if len(self.buffer) > batch_size else len(self.buffer)
        return random.sample(self.buffer, size)

    def clear(self):
        self.buffer.clear()

    def append(self, trainsition):
        self.buffer.append(trainsition)

    def __len__(self):
        return len(self.buffer)