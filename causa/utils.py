from math import ceil
import torch


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch
    sampling and adaptive augmentation on GPU.
    """

    def __init__(self, x, y, transform=None, transform_y=None, batch_size=500, 
                 data_factor=1, shuffle=False):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.data_factor = data_factor
        self.n_data = y.size(0)
        self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle
        identity = lambda x: x
        self.transform = transform if transform is not None else identity
        self.transform_y = transform_y if transform_y is not None else identity

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration
        
        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        batch = (self.transform(self.x[start:end]), self.transform_y(self.y[start:end]))
        self.i_batch += 1
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def dataset(self):
        return DatasetDummy(self.n_data * self.data_factor)


class DatasetDummy:
    def __init__(self, N):
        self.N = N
        
    def __len__(self):
        return self.N
