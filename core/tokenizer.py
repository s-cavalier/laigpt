import numpy as np
from abc import ABCMeta, abstractmethod

class Tokenizer(metaclass=ABCMeta):    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        pass

    @abstractmethod
    def __getitem__(self, token: str) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class CharTokenizer(Tokenizer):
    def __init__(self, characters: str):
        chars = sorted(set(characters))

        self.UNK = "<UNK>"
        if self.UNK not in chars:
            chars.append(self.UNK)

        self.token_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_token = [c for c in chars]

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_idx[self.UNK]
        return [self.token_to_idx.get(c, unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.idx_to_token[i] for i in ids)

    def keys(self) -> list[str]:
        return list(self.token_to_idx.keys())

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx[token]

    def __len__(self):
        return len(self.idx_to_token)


class WordTokenizer(Tokenizer):
    def __init__(self, text: str):
        words = sorted(set(text.split()))
        self.UNK = "<UNK>"

        if self.UNK not in words:
            words.append(self.UNK)

        self.word_to_idx = {w: i for i, w in enumerate(words)}
        self.idx_to_word = [w for w in words]

    def encode(self, text: str) -> list[int]:
        unk = self.word_to_idx[self.UNK]
        return [self.word_to_idx.get(w, unk) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return ' '.join(self.idx_to_word[i] for i in ids)

    def keys(self) -> list[str]:
        return list(self.word_to_idx.keys())

    def __getitem__(self, token: str) -> int:
        return self.word_to_idx[token]

    def __len__(self):
        return len(self.idx_to_word)
