import pickle
from random import shuffle


class Candidate():
    def __init__(self, origin_data: dict, encoded_data: list, statistics_label=None, acc_label=None,
                 acc_each_epoch=None):
        self.o_data = origin_data
        self.e_data = encoded_data
        self.statistics_label = statistics_label
        self.acc_label = acc_label
        self.acc_each_epoch = acc_each_epoch

        self.predict_acc = None

    def __str__(self):
        return '[o_data: {}, '.format(self.o_data) +\
               'e_data: {}, '.format(self.e_data) +\
               'statistics_label: {}, '.format(self.statistics_label) +\
               'acc_label: {}, '.format(self.acc_label) +\
               'acc_each_epoch: {}, '.format(self.acc_each_epoch) +\
               'predict_acc: {}]'.format(self.predict_acc)


class CandidatePool():
    # pool: [candidate1, candidate2, ...]
    def __init__(self, pool=list(), deduplication=True):
        self.__deduplication = deduplication
        self.__encoded_set = set()

        if self.__check(pool):
            self.__pool = self.__deduplicate_pool(pool)
        else:
            raise TypeError

        self.eval_pool = list()

    def __len__(self):
        return len(self.__pool)

    def __getitem__(self, key):
        return self.__pool[key]

    def __check(self, candidates):
        if not isinstance(candidates, list):
            return False
        else:
            for candidate in candidates:
                if not isinstance(candidate, Candidate):
                    raise TypeError('Element with type {}, need Candidate'.format(type(candidate)))
            return True

    def __deduplicate_pool(self, pool: list):
        if self.__deduplication:
            new_pool = list()
            for candidate in pool:
                if tuple(candidate.e_data) not in self.__encoded_set:
                    self.__encoded_set.add(tuple(candidate.e_data))
                    new_pool.append(candidate)
            return new_pool
        else:
            return pool

    def __check_duplicate(self, candidate: Candidate):
        if tuple(candidate.e_data) in self.__encoded_set:
            return False
        else:
            return True

    def append(self, candidate: Candidate):
        if isinstance(candidate, Candidate):
            if self.__deduplication:
                if self.__check_duplicate(candidate):
                    self.__encoded_set.add(tuple(candidate.e_data))
                    self.__pool.append(candidate)
            else:
                self.__pool.append(candidate)
        else:
            raise TypeError

    def extend(self, candidates: list):
        if self.__check(candidates):
            self.__pool.extend(self.__deduplicate_pool(candidates))
        else:
            raise TypeError

    def pending_eval(self, m: int, random=False):
        if random:
            shuffle(self.__pool)
            return self.__pool[:m]
        else:
            self.__pool.sort(key=lambda candidate: candidate.predict_acc, reverse=True)
            pending_pool = list()
            for candidate in self.__pool:
                if candidate.acc_label is None:
                    pending_pool.append(candidate)
                    if len(pending_pool) == m:
                        break
            return pending_pool

    def topK(self, k: int):
        self.__pool.sort(key=lambda candidate: candidate.predict_acc, reverse=True)
        return self.__pool[:k]

    def receive_candidate(self, candidate: Candidate):
        assert candidate.acc_label is not None
        self.eval_pool.append(candidate)


# idx: the range of idx is [0, class_num - 1]
def one_hot(idx: int, class_num: int):
    res = [0 for _ in range(class_num)]
    res[idx] = 1
    return res

def deepcopy(ob):
    return pickle.loads(pickle.dumps(ob))


if __name__ == "__main__":
    import random

    candidates = [Candidate({}, [i, i]) for i in range(10)]
    candidate_pool = CandidatePool(candidates, deduplication=True)

    for candidate in candidate_pool:
        print(candidate)
        candidate.predict_acc = random.random()
    print('=======================================================')
    candidate_pool.append(Candidate({}, [2, 2]))
    for candidate in candidate_pool:
        print(candidate)
    print('=======================================================')
    print(max(candidate_pool, key=lambda k: k.predict_acc))
    print('=======================================================')
    for candidate in candidate_pool.topK(3):
        print(candidate)
