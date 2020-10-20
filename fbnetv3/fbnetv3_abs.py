from abc import ABC, abstractclassmethod

from fbnetv3_utils import Candidate, CandidatePool


class AbsSearchSpaceCodec(ABC):
    @abstractclassmethod
    def encode(self, origin_data: dict) -> list:
        pass

    @abstractclassmethod
    def decode(self, encoded_data: list) -> dict:
        pass

    @abstractclassmethod
    def random_generate(self) -> Candidate:
        pass


class AbsGeneticAlgorithm(ABC):
    @abstractclassmethod
    def cross(self, population: list, enlarge_rate: int) -> list:
        pass


class AbsUnderEvalNet(ABC):
    @abstractclassmethod
    def __init__(self, candidate: Candidate):
        pass

    @abstractclassmethod
    def set_statistics_label(self):
        pass

    @abstractclassmethod
    def set_acc_label(self, total_epoch: int, record_acc_each_epoch=False):
        pass
