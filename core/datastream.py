from abc import ABCMeta, abstractmethod
import numpy as np

class DataSource(metaclass=ABCMeta):
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_slice(self, start: int, end: int) -> np.ndarray:
        pass

class DataStream(metaclass=ABCMeta):
    def __init__(self, sink: DataSource, spout: DataSource | None = None, batch_size: int = 1 ):
        self.sink = sink
        self.spout = spout if spout is not None else sink
        self.batch_size = batch_size

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def __len__(self) -> int:
        return int(np.ceil(min(len(self.sink), len(self.spout)) / self.batch_size))

# Some implmentations

class NumpyArraySource(DataSource):
    def __init__(self, arr: np.ndarray): self.arr = arr
    def __len__(self): return len(self.arr)
    def get_slice(self, start: int, end: int): return self.arr[start:end]

class TextSource(DataSource):
    def __init__(self, ids: np.ndarray):
        assert ids.ndim == 1, "TextSource expects a 1-D array of token IDs"
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def get_slice(self, start: int, end: int) -> np.ndarray:
        return self.ids[start:end]


class MNISTData(DataStream):

    def __init__(self, img_src: DataSource, label_src: DataSource, batch_size : int, shuffle=True):
        super().__init__(img_src, label_src, batch_size)
        self.shuffle = shuffle
        self.indicies = np.arange( len(img_src) )
        self.cursor = 0

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.indicies)
        self.cursor = 0
        return self
    
    def __next__(self):
        if self.cursor >= len(self.indicies): raise StopIteration

        idx = self.indicies[ self.cursor : self.cursor + self.batch_size ]
        self.cursor += self.batch_size

        X = np.stack([self.sink.get_slice(i, i + 1)[0] for i in idx])
        Y = np.stack([self.spout.get_slice(i, i + 1)[0] for i in idx])
        return X, Y

class GPTData(DataStream):
    """
    DataStream for GPT text training.

    Given a 1-D sequence of token IDs, yields:
        X: input sequence of length seq_len
        Y: next-token targets of length seq_len
    """

    def __init__(
        self,
        token_source: DataSource, 
        seq_len: int,
        batch_size: int,
        shuffle: bool = True
    ):
        super().__init__(token_source, token_source, batch_size)
        self.seq_len = seq_len
        self.shuffle = shuffle

        self.n_tokens = len(token_source)
        self.n_windows = self.n_tokens - seq_len

        self.indices = np.arange(self.n_windows)
        self.cursor = 0

    def __len__(self):
        return int(np.ceil(self.n_windows / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= self.n_windows:
            raise StopIteration

        start_batch = self.cursor
        end_batch = self.cursor + self.batch_size
        idxs = self.indices[start_batch:end_batch]
        self.cursor += self.batch_size

        X = []
        Y = []

        for start in idxs:
            x = self.sink.get_slice(start, start + self.seq_len)
            y = self.spout.get_slice(start + 1, start + 1 + self.seq_len)

            X.append(x)
            Y.append(y)

        return np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)
