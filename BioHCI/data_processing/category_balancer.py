from abc import ABC, abstractmethod


class CategoryBalancer(ABC):

    @abstractmethod
    def balance(self, compacted_subj_dict):
        pass

