import random
import pickle
from functools import reduce

from fbnetv3_abs import AbsGeneticAlgorithm
from fbnetv3_utils import Candidate, CandidatePool, deepcopy


class AdaptiveGeneticAlgorithm(AbsGeneticAlgorithm):
    def __init__(self, cross_p_min: float, cross_p_max: float, max_epoch: int, encode_func):
        assert cross_p_max > cross_p_min
        self.cross_p_min = cross_p_min
        self.cross_p_max = cross_p_max
        self.cross_p_gap = cross_p_max - cross_p_min
        self.max_epoch = max_epoch
        self.curr_epoch = 0

        self.encode_func = encode_func

        self.max_acc = 0
        self.avg_acc = 0
        self.acc_gap = 0

    def cross(self, population: CandidatePool, children_rate: int):
        assert len(population) > 0

        self.curr_epoch = self.curr_epoch if self.curr_epoch == self.max_epoch else (self.curr_epoch + 1)
        mutate_rate = 1 - random.random() ** (1 - self.curr_epoch / self.max_epoch)
        self.max_acc = max(population, key=lambda individual: individual.predict_acc).predict_acc[0]
        if len(population) > 1:
            self.avg_acc = sum([individual.predict_acc[0] for individual in population]) / len(population)
        else:
            self.avg_acc = population[0].predict_acc[0]
        self.acc_gap = self.max_acc - self.avg_acc

        child_population = list()
        for individual_a in population:
            for _ in range(children_rate):
                individual_b = random.choice(population)
                cross_p = self.__cross_p(individual_a.predict_acc[0], individual_b.predict_acc[0])
                new_individual = self.__cross_individual(individual_a, individual_b, cross_p)
                self.__mutate(new_individual)
                child_population.append(new_individual)
        return child_population

    def __cross_individual(self, individual_a: Candidate, individual_b: Candidate, cross_p: float):
        new_o_data = dict()
        a_o_data = individual_a.o_data
        b_o_data = individual_b.o_data
        for stage_name, op_dict in a_o_data.items():
            new_o_data[stage_name] = dict()
            if stage_name == 'recipe':
                for key, val in op_dict.items():
                    if key in b_o_data[stage_name] and random.random() < cross_p:
                        new_o_data[stage_name][key] = deepcopy(b_o_data[stage_name][key])
                    else:
                        new_o_data[stage_name][key] = deepcopy(val)
            else:
                layer_name_a, params_a = list(op_dict.items())[0]
                layer_name_b, params_b = list(b_o_data[stage_name].items())[0]
                if layer_name_a == layer_name_b:
                    new_o_data[stage_name][layer_name_a] = dict()
                    for key, val in params_a.items():
                        if key in params_b and random.random() < cross_p:
                            new_o_data[stage_name][layer_name_a][key] = deepcopy(params_b[key])
                        else:
                            new_o_data[stage_name][layer_name_a][key] = deepcopy(val)
                elif random.random() < cross_p:
                    new_o_data[stage_name][layer_name_b] = deepcopy(params_b)
                else:
                    new_o_data[stage_name][layer_name_a] = deepcopy(params_a)
        return Candidate(new_o_data, self.encode_func(new_o_data))

    def __cross_p(self, acc_a, acc_b):
        acc = max(acc_a, acc_b)
        if acc < self.avg_acc:
            return self.cross_p_max
        else:
            return self.cross_p_max - self.cross_p_gap * (acc - self.avg_acc) / self.acc_gap

    def __mutate(self, individual: dict):
        return individual
