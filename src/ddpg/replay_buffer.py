import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = -1
        self.data = []

    def add(self, entry):
        if len(self.data) < self.capacity:
            self.idx += 1
            self.data.append(entry)
        else:
            self.idx = (self.idx + 1) % self.capacity
            self.data[self.idx] = entry

    def sample_batch(self, batch_size):
        if len(self.data) < batch_size:
            raise Exception(f"Not enough data in replay buffer: {len(self.data)} < {batch_size}")

        batch = random.sample(self.data, batch_size)
        tuple_len = len(batch[0])

        return [torch.stack([entry[i] for entry in batch], dim=0) for i in range(tuple_len)]
