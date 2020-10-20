import random

from fbnetv3_abs import AbsSearchSpaceCodec
from fbnetv3_utils import one_hot, Candidate


class SearchSpaceCodec(AbsSearchSpaceCodec):
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

        self.__arch_input_size = len(self.__layer_names)
        self.__recipe_input_size = 0
        for param_range in self.__layer_param_range:
            self.__arch_input_size += len(param_range) if isinstance(param_range, list) else 1
        self.__arch_input_size *= len(self.__stage_names)
        for param_range in self.__hyperparam_range:
            self.__recipe_input_size += len(param_range) if isinstance(param_range, list) else 1

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
        return Candidate(origin_data, self.encode(origin_data))

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

    def encode(self, origin_data: dict):
        encoded_arch_data = list()
        encoded_recipe_data = list()
        for stage_name in self.__stage_names:
            layer_name, layer_params = list(origin_data[stage_name].items())[0]
            encoded_arch_data.extend(one_hot(self.__layer_names.index(layer_name), len(self.__layer_names)))
            for idx, param_name in enumerate(self.__layer_param_names, 0):
                param_range = self.__layer_param_range[idx]
                if param_name not in layer_params or layer_params[param_name] is None:
                    encoded_arch_data.extend([0 for _ in range(len(param_range) if isinstance(param_range, list) else 1)])
                else:
                    encoded_arch_data.extend(self.__encode_choice_val(layer_params[param_name], param_range))
        hyperparam = origin_data['recipe']
        for idx, param_name in enumerate(self.__hyperparam_names, 0):
            param_range = self.__hyperparam_range[idx]
            if param_name not in hyperparam or hyperparam[param_name] is None:
                encoded_recipe_data.extend([0 for _ in range(len(param_range) if isinstance(param_range, list) else 1)])
            else:
                encoded_recipe_data.extend(self.__encode_choice_val(hyperparam[param_name], param_range))
        return encoded_arch_data + encoded_recipe_data

    def __encode_choice_val(self, choice_val, param_range):
        if isinstance(param_range, tuple):
            return [(choice_val - param_range[0]) / (param_range[1] - param_range[0])]
        elif isinstance(param_range, list):
            return one_hot(param_range.index(choice_val), len(param_range))
        else:
            raise TypeError()

    def decode(self, encoded_data: list):
        pass

    def arch_recipe_size(self):
        return self.__arch_input_size, self.__recipe_input_size
