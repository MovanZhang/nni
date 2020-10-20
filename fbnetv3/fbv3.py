import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.utils.data import DataLoader, Dataset

from ptflops import get_model_complexity_info


class Candidate():
    def __init__(self, origin_data: dict, encoded_data, statistics_label=None, acc_label=None):
        self.o_data = origin_data
        self.e_data = encoded_data
        self.statistics_label = statistics_label
        self.acc_label = acc_label

        self.predict_acc = None


class CandidatePool():
    # pool: [candidate1, candidate2, ...]
    # predict_func: is the function that predict accuracy for candidate and modify candidate.predict_acc
    def __init__(self, pool=None, predict_func=None):
        if pool is None or len(pool) == 0:
            self.__pool = list()
        elif self.__check(pool):
            self.__pool = pool
        else:
            raise TypeError

        self.__predict = predict_func
        self.__eval_pool = list()

    def __len__(self):
        return len(self.__pool)

    def __check(self, candidates):
        if not isinstance(candidates, list):
            return False
        else:
            for candidate in candidates:
                if not isinstance(candidate, Candidate):
                    raise TypeError('Element with type {}, need Candidate'.format(type(candidate)))
            return True

    def __predict_all(self):
        assert self.__predict is not None
        for candidate in self.__pool:
            self.__predict(candidate)

    def append(self, candidate: Candidate):
        if isinstance(candidate, Candidate):
            self.__pool.append(candidate)
        else:
            raise TypeError

    def extend(self, candidates: list):
        if self.__check(candidates):
            self.__pool.extend(candidates)
        else:
            raise TypeError

    def pending_eval(self, m: int):
        self.__predict_all()
        self.__pool.sort(key=lambda candidate: candidate.predict_acc, reverse=True)
        pending_pool = list()
        for candidate in self.__pool:
            if candidate.acc_label is None:
                pending_pool.append(candidate)
                if len(pending_pool) == m:
                    break
        return pending_pool

    def topK(self, k: int):
        self.__predict_all()
        self.__pool.sort(key=lambda candidate: candidate.predict_acc, reverse=True)
        return self.__pool[:k]

    def receive_candidate(self, candidate: Candidate):
        assert candidate.acc_label is not None
        self.__eval_pool.append(candidate)

    def set_predict_func(self, predict_func):
        self.__predict = predict_func


class CandidatePool_v1():
    def __init__(self, predictor: nn.Module, origin_pool: list, encoded_pool: Tensor):
        self.predictor = predictor
        assert len(origin_pool) == encoded_pool.size()[0]
        self.origin_pool = origin_pool
        self.encoded_pool = encoded_pool
        self.predict_accuracy = None

        self.evaluated_candidates = Tensor()
        self.evaluated_index_set = set()
        self.evaluated_accuracy = Tensor()

    def append_pool(self, origin_data, encoded_data):
        self.origin_pool.append(origin_data)
        self.encoded_pool = torch.cat((self.encoded_pool, torch.unsqueeze(Tensor(encoded_data), dim=0)), dim=0)

    def add_eval_data(self, candidates_idx_list: list, eval_acc: Tensor):
        assert len(candidates_idx_list) == eval_acc.size()[0]
        new_evaluated_candidates = self.encoded_pool.index_select(0, torch.tensor(candidates_idx_list, dtype=torch.int64))
        self.evaluated_candidates = torch.cat((self.evaluated_candidates, new_evaluated_candidates), dim=0)
        self.evaluated_index_set.update(candidates_idx_list)
        self.evaluated_accuracy = torch.cat((self.evaluated_accuracy, eval_acc), dim=0)

    def get_candidates(self, m: int):
        self.__predict_all()
        _, indices = torch.sort(self.predict_accuracy, descending=True)
        candidates_idx_list = list()
        idx = 0
        while len(candidates_idx_list) < m and idx < len(self.encoded_pool):
            if indices[idx] not in self.evaluated_index_set:
                candidates_idx_list.append(indices[idx])
            idx += 1
        return candidates_idx_list, [self.origin_pool[idx] for idx in candidates_idx_list], self.encoded_pool.index_select(0, torch.tensor(candidates_idx_list, dtype=torch.int64))

    def get_top_p(self, p: int):
        self.__predict_all()
        _, indices = torch.sort(self.predict_accuracy, descending=True)
        candidates_idx_list = indices[:p]
        origin_data_list = [self.origin_pool[idx] for idx in candidates_idx_list]
        acc_list = torch.squeeze(self.predict_accuracy.index_select(0, torch.tensor(candidates_idx_list, dtype=torch.int64))).tolist()
        return origin_data_list, acc_list

    def get_acc_train_data(self):
        return self.evaluated_candidates, self.evaluated_accuracy

    # def __predict(self, idx: int) -> float:
    #     idx = len(self.predict_accuracy) + idx if idx < 0 else idx
    #     self.predictor.eval()
    #     self.predict_accuracy = torch.cat(self.predict_accuracy, self.predictor(self.encoded_pool[idx:idx + 1], mode='accuracy'))
    #     self.predictor.train()

    def __predict_all(self):
        self.predictor.eval()
        predict_accuracy = [self.predictor(self.encoded_pool[idx:idx + 1], mode='accuracy').item()
                            for idx in range(len(self.encoded_pool))]
        self.predict_accuracy = Tensor(predict_accuracy)
        self.predictor.train()


class SearchSpaceCodec():
    def __init__(self, search_space: dict):
        self.search_space = search_space

        self.__stage_names = list()
        self.__layer_names = list()
        self.__layer_param_names = list()
        self.__layer_param_range = list()
        self.__hyperparam_names = list()
        self.__hyperparam_range = list()

        for stage_name, choice in self.search_space.items():
            if choice['_type'] == 'layer_choice':
                self.__stage_names.append(stage_name)
                for layer_name, layer_params in choice['_value']:
                    if layer_name not in self.__layer_names:
                        self.__layer_names.append(layer_name)
                    for param_name, param_range in layer_params.items():
                        if param_range is not None and param_name not in self.__layer_param_names:
                            self.__layer_param_names.append(param_name)
                            if isinstance(param_range, list):
                                self.__layer_param_range.append(list(param_range))
                            elif isinstance(param_range, tuple):
                                param_range = param_range[0:2] if len(param_range) > 1 else (param_range[0], param_range[0])
                                self.__layer_param_range.append(param_range)
                            else:
                                raise KeyError('{}: {}'.format(param_name, param_range))
                        elif param_name in self.__layer_param_names and isinstance(param_range, list):
                            idx = self.__layer_param_names.index(param_name)
                            self.__layer_param_range[idx].extend(param_range)
                        elif param_name in self.__layer_param_names and isinstance(param_range, tuple):
                            idx = self.__layer_param_names.index(param_name)
                            param_range = param_range[0:2] if len(param_range) > 1 else (param_range[0], param_range[0])
                            self.__layer_param_range[idx] = (min(param_range[0], self.__layer_param_range[idx][0]),
                                                             max(param_range[1], self.__layer_param_range[idx][1]))
                        else:
                            assert param_range is None or type(param_range) is tuple
            elif choice['_type'] == 'recipe_choice':
                for hp_name, hp_range in choice['_value'].items():
                    self.__hyperparam_names.append(hp_name)
                    if isinstance(hp_range, list):
                        self.__hyperparam_range.append(list(hp_range))
                    elif isinstance(hp_range, tuple):
                        hp_range = hp_range[0:2] if len(hp_range) > 1 else (hp_range[0], hp_range[0])
                        self.__hyperparam_range.append(hp_range)
                    else:
                        raise KeyError('{}: {}'.format(hp_name, hp_range))

        for idx, param_range in enumerate(self.__layer_param_range, 0):
            if isinstance(param_range, list):
                self.__layer_param_range[idx] = list(set(param_range))

        self.arch_input_size = len(self.__layer_names)
        self.recipe_input_size = 0
        for param_range in self.__layer_param_range:
            self.arch_input_size += len(param_range) if isinstance(param_range, list) else 1
        self.arch_input_size *= len(self.__stage_names)
        for param_range in self.__hyperparam_range:
            self.recipe_input_size += len(param_range) if isinstance(param_range, list) else 1

    def encode(self, origin_data: dict):
        encoded_arch_data = Tensor()
        encoded_recipe_data = Tensor()
        for stage_name in self.__stage_names:
            layer_name, layer_params = list(origin_data[stage_name].items())[0]
            encoded = F.one_hot(torch.tensor([self.__layer_names.index(layer_name)], dtype=torch.int64), len(self.__layer_names))[0]
            encoded_arch_data = torch.cat((encoded_arch_data, encoded))
            for idx, param_name in enumerate(self.__layer_param_names, 0):
                if param_name not in layer_params or layer_params[param_name] is None:
                    encoded_arch_data = torch.cat((encoded_arch_data, torch.zeros((len(self.__layer_param_range[idx]), ))
                                                   if isinstance(self.__layer_param_range[idx], list) else torch.zeros((1, ))))
                else:
                    encoded = self.__encode_choice_val(layer_params[param_name], self.__layer_param_range[idx])
                    encoded_arch_data = torch.cat((encoded_arch_data, encoded))
        hyperparam = origin_data['recipe']
        for idx, param_name in enumerate(self.__hyperparam_names, 0):
            if param_name not in hyperparam or hyperparam[param_name] is None:
                encoded_recipe_data = torch.cat((encoded_recipe_data, torch.zeros((len(self.__hyperparam_range[idx]), ))))
            else:
                encoded = self.__encode_choice_val(hyperparam[param_name], self.__hyperparam_range[idx])
                encoded_recipe_data = torch.cat((encoded_recipe_data, encoded))
        return torch.unsqueeze(torch.cat((encoded_arch_data, encoded_recipe_data)), dim=0)

    def __encode_choice_val(self, choice_val, param_range):
        if isinstance(param_range, tuple):
            return Tensor([(choice_val - param_range[0]) / (param_range[1] - param_range[0])])
        elif isinstance(param_range, list):
            return F.one_hot(torch.tensor([param_range.index(choice_val)]), len(param_range))[0]
        else:
            raise TypeError()

    def decode(self, encoded_data: Tensor):
        pass

    def random_generate(self):
        origin_data = dict()
        for stage_name in self.__stage_names:
            layer_name, layer_params = random.choice(self.search_space[stage_name]['_value'])
            origin_data[stage_name] = {layer_name: {}}
            for param_name, param_range in layer_params.items():
                if param_range is not None:
                    choice_val = self.__random_choice(param_range)
                    origin_data[stage_name][layer_name][param_name] = choice_val
        origin_data['recipe'] = dict()
        for param_name, param_range in self.search_space['recipe']['_value'].items():
            choice_val = self.__random_choice(param_range)
            origin_data['recipe'][param_name] = choice_val
        return origin_data

    def __random_choice(self, param_range):
        if isinstance(param_range, tuple):
            if len(param_range) == 1:
                return param_range[0]
            elif len(param_range) == 2:
                return random.randint(param_range[0], param_range[1])
            elif len(param_range) == 3:
                return random.randrange(param_range[0], param_range[1] + 1, param_range[2])
        elif isinstance(param_range, list):
            assert len(param_range) > 0
            return random.choice(param_range)
        else:
            raise KeyError('range: {} is invalid'.format(param_range))


class FBNetV3():
    def __init__(self, search_space: dict, pretrain_sample_size: int, eval_module: nn.Module, train_eval_module,
                 aga, predictor_module=FBNetV3_Predictor):
        self.search_space = search_space
        self.eval_module = eval_module
        self.train_eval_module = train_eval_module
        self.ss_codec = SearchSpaceCodec(self.search_space)
        self.predictor = predictor_module(self.ss_codec.arch_input_size, self.ss_codec.recipe_input_size)
        self.aga = aga
        self.__pretrain_embedding_layer(pretrain_sample_size)

    def generate_parameters(self, parameter_id):
        return self.ss_codec.random_generate()

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        pass

    # how many arch to sample? train epoch?
    def __pretrain_embedding_layer(self, data_size: int):
        split = int(data_size * 0.8)
        data = Tensor()
        labels = Tensor()
        # for i in range(1200):
        #     origin_data = self.ss_codec.random_generate()
        #     data = torch.cat((data, self.ss_codec.encode(origin_data)), dim=0)
        #     labels = torch.cat((labels, torch.unsqueeze(Tensor(get_model_complexity_info(self.eval_module(origin_data),
        #                         (3, 32, 32), as_strings=False, print_per_layer_stat=False)) / 1000000, dim=0)), dim=0)
        #     print(i)
        origin_data_list = [self.ss_codec.random_generate() for _ in range(data_size)]
        data = torch.cat([self.ss_codec.encode(origin_data) for origin_data in origin_data_list], dim=0)
        labels = Tensor([get_model_complexity_info(self.eval_module(origin_data), (3, 32, 32),
                         as_strings=False, print_per_layer_stat=False) for origin_data in origin_data_list]) / 10000
        train_dataset = SampleDataset(data[:split], labels[:split])
        trian_dataloader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
        test_dataset = SampleDataset(data[split:], labels[split:])
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=50, shuffle=True)

        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(self.predictor.parameters(), lr=0.001, momentum=0.9)
        print('[Proxy Predictor] Begin training embeddeding layer...')
        for i in range(10):
            total_loss = 0
            self.predictor.train()
            for data, labels in trian_dataloader:
                optimizer.zero_grad()
                outputs = self.predictor(data, mode='proxy')
                loss = criterion(outputs, labels)
                total_loss += loss
                loss.backward()
                optimizer.step()
            print('[Proxy Predictor] {:3d} epoch with train-loss: {}'.format(i, total_loss / split))

            total_loss = 0
            self.predictor.eval()
            for data, labels in test_dataloader:
                outputs = self.predictor(data, mode='proxy')
                loss = criterion(outputs, labels)
                total_loss += loss
            print('[Proxy Predictor] {:3d} epoch with test-loss: {}'.format(i, total_loss / (data_size - split)))
        print('[Proxy Predictor] End training embeddeding layer...')

    def __QMC_sampling_pool(self, pool_size: int):
        # need to change to QMC
        origin_pool = list()
        encoded_pool = Tensor()
        i = 0
        while i < pool_size:
            origin_data = self.ss_codec.random_generate()
            encoded_data = self.ss_codec.encode(origin_data)
            if origin_data not in origin_pool:
                origin_pool.append(origin_data)
                encoded_pool = torch.cat((encoded_pool, encoded_data), dim=0)
                i += 1
            else:
                print('find conflict')
        self.__initialize_candidate_pool(origin_pool, encoded_pool)

    def __initialize_candidate_pool(self, origin_data, encoded_data):
        self.pool = CandidatePool(self.predictor, origin_data, encoded_data)

    def __acc_predictor_train_step(self, m: int):
        idx_list, origin_data_list, encoded_data_list = self.pool.get_candidates(m=m)
        eval_acc = list()
        for args in origin_data_list:
            print('Parameter: {}'.format(args))
            eval_acc.append(self.train_eval_module(args))
        self.pool.add_eval_data(idx_list, torch.unsqueeze(torch.tensor(eval_acc), dim=1))
        print(self.pool.get_acc_train_data())

        data, labels = self.pool.get_acc_train_data()
        dataset = SampleDataset(data, labels)
        dataloader = DataLoader(dataset=dataset, batch_size=m, shuffle=True)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(self.predictor.parameters(), lr=0.001, momentum=0.9)
        print('[ACC Predictor] Begin training embeddeding layer...')
        for i in range(20):
            total_loss = 0
            self.predictor.train()
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.predictor(data, mode='accuracy')
                loss = criterion(outputs, labels)
                total_loss += loss
                loss.backward()
                optimizer.step()
            print('[ACC Predictor] {:3d} epoch with train-loss: {}'.format(i, total_loss / len(dataset)))
        print('[ACC Predictor] End training embeddeding layer...')

    def __evolutionary_search_step(self, origin_data_list: list, acc_list: list, k: int):
        children = self.aga.cross(origin_data_list, acc_list)
        for child in children:
            if child not in origin_data_list:
                origin_data_list.append(child)
        acc_list = self.__predict_all(self.__encode_all(origin_data_list))
        indices = np.array(acc_list).argsort()[-k:][::-1]

        new_origin_data_list = [origin_data_list[idx] for idx in indices]
        new_acc_list = [acc_list[idx] for idx in indices]

        return new_origin_data_list, new_acc_list

    def __predict_all(self, encoded_data_list):
        self.predictor.eval()
        predict_accuracy_list = [self.predictor(encoded_data_list[idx], mode='accuracy').item()
                                 for idx in range(len(encoded_data_list))]
        self.predictor.train()
        print(predict_accuracy_list)
        return predict_accuracy_list

    def __encode_all(self, origin_data_list):
        encoded_data_list = [self.ss_codec.encode(data) for data in origin_data_list]
        return encoded_data_list

    def run(self, n: int, m: int, T: int, p: int, q: int, e: float):
        self.__QMC_sampling_pool(pool_size=n)

        for t in range(T):
            self.__acc_predictor_train_step(m)

        origin_data_list, _ = self.pool.get_top_p(p=p)
        while len(origin_data_list) < p + q:
            data = self.ss_codec.random_generate()
            if data not in origin_data_list:
                origin_data_list.append(data)
            else:
                print('find conflict')
        predict_accuracy_list = self.__predict_all(self.__encode_all(origin_data_list))
        s = max(predict_accuracy_list)
        s0 = 0

        while abs(s - s0) > e:
            s0 = s
            origin_data_list, predict_accuracy_list = self.__evolutionary_search_step(origin_data_list, predict_accuracy_list, p + q)

        print(origin_data_list)
        print(predict_accuracy_list)


if __name__ == "__main__":
    from search_net import SearchNet, train_search_net
    from test_utils import search_space_v2
    from aga import AdaptiveGeneticAlgorithm

    aga = AdaptiveGeneticAlgorithm(0, 0.5, 10)

    fbnetv3 = FBNetV3(search_space_v2, 5000, SearchNet, train_search_net, aga)
    fbnetv3.run(n=10000, m=5, T=3, p=20, q=20, e=0.000001)

    # origin_data = fbnetv3.ss_codec.random_generate()
    # print(origin_data)
    # encoded_data = fbnetv3.ss_codec.encode(origin_data)
    # print(encoded_data)
