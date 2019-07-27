from abc import ABC, abstractmethod


class CategoryBalancer(ABC):

    @abstractmethod
    def balance(self, subj_dict):
        pass

