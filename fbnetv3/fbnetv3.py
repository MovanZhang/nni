from fbnetv3_abs import AbsSearchSpaceCodec, AbsGeneticAlgorithm, AbsUnderEvalNet
from fbnetv3_predictor import Predictor
from fbnetv3_utils import Candidate, CandidatePool


class FBNetV3():
    def __init__(self, predictor: Predictor, ss_codec: AbsSearchSpaceCodec, genetic_algorithm: AbsGeneticAlgorithm):
        self.predictor = predictor
        self.codec = ss_codec
        self.evolutionary = genetic_algorithm

    def run(self, pre_train_data_size: int, pool_size: int, iter_batch_size: int, iter_batch_num: int,
            UnderEvalNet: AbsUnderEvalNet, eval_total_epoch: int, eval_early_stop_threshold: float, best_p: int,
            random_q: int, enlarge_rate: int):
        # pre-train embedding layer
        self.__pretrain_embedding_layer(pre_train_data_size, UnderEvalNet)

        # constrained iterative optimization
        candidate_pool = self.__QMC_sampling_pool(pool_size)

        # self.predictor.predict_all(candidate_pool)
        candidates = candidate_pool.pending_eval(iter_batch_size, random=True)
        for candidate in candidates:
            UnderEvalNet(candidate).set_acc_label(total_epoch=eval_total_epoch, record_acc_each_epoch=True)
            candidate_pool.receive_candidate(candidate)
        early_stop_epoch = self.__compute_early_stop_epoch(candidates, eval_early_stop_threshold)
        self.predictor.train_predictor(candidate_pool.eval_pool, mode='accuracy',
                                       dataset_split_rate=1, batch_size=5, epoch=10)

        for _ in range(1, iter_batch_num):
            self.predictor.predict_all(candidate_pool)
            candidates = candidate_pool.pending_eval(iter_batch_size)
            for candidate in candidates:
                UnderEvalNet(candidate).set_acc_label(total_epoch=early_stop_epoch, record_acc_each_epoch=False)
                candidate_pool.receive_candidate(candidate)
            self.predictor.train_predictor(candidate_pool.eval_pool, mode='accuracy',
                                           dataset_split_rate=1, batch_size=5, epoch=10)

        # predictor_based evolutionary search
        population = CandidatePool(candidate_pool.topK(best_p), deduplication=True)
        while len(population) < best_p + random_q:
            population.append(self.codec.random_generate())
        self.predictor.predict_all(population)

        s = max(population, key=lambda individual: individual.predict_acc).predict_acc[0]
        s0 = 0
        # hard code
        cnt = 0
        while abs(s - s0) > 0.000001:
            population.extend(self.evolutionary.cross(population, enlarge_rate))
            self.predictor.predict_all(population)
            population = CandidatePool(population.topK(best_p + random_q), deduplication=True)
            s0 = s
            s = max(population, key=lambda individual: individual.predict_acc).predict_acc[0]

        for individual in population:
            print(individual, flush=True)

    def __pretrain_embedding_layer(self, data_size: int, UnderEvalNet: AbsUnderEvalNet):
        candidates = self.__no_repeat_sampling_pool(data_size)
        for candidate in candidates:
            UnderEvalNet(candidate).set_statistics_label()
        # hard code
        self.predictor.train_predictor(candidates, mode='proxy', dataset_split_rate=0.8, batch_size=100, epoch=10)

    # hard code
    def __compute_early_stop_epoch(self, candidates: list, eval_early_stop_threshold: float):
        return len(candidates[0].acc_each_epoch)

    # need to support QMC sampling
    def __QMC_sampling_pool(self, pool_size: int) -> CandidatePool:
        return self.__no_repeat_sampling_pool(pool_size)

    def __no_repeat_sampling_pool(self, pool_size: int) -> CandidatePool:
        pool = CandidatePool(deduplication=True)
        while len(pool) < pool_size:
            pool.append(self.codec.random_generate())
        return pool


if __name__ == "__main__":
    import sys

    from fbnetv3_predictor import Predictor
    from fbnetv3_codec import SearchSpaceCodec
    from fbnetv3_aga import AdaptiveGeneticAlgorithm
    from fbnetv3_usnet import UnderEvalNet

    from test_utils import search_space_v2 as search_space

    o_stdout = sys.stdout
    o_stderr = sys.stderr

    with open('out.txt', 'w+') as f_out:
        with open('err.txt', 'w+') as f_err:
            sys.stdout = f_out
            sys.stderr = f_err

            codec = SearchSpaceCodec(search_space)
            arch_size, recipe_size = codec.arch_recipe_size()
            predictor = Predictor(arch_size, recipe_size)
            aga = AdaptiveGeneticAlgorithm(0, 0.5, 20, codec.encode)

            fbnetv3 = FBNetV3(predictor, codec, aga)
            fbnetv3.run(10000, 100000, 40, 4, UnderEvalNet, 2, 1, 40, 40, 6)

    sys.stdout = o_stdout
    sys.stderr = o_stderr
