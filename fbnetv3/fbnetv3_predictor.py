import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim, Tensor
from torch.utils.data import DataLoader, Dataset

from fbnetv3_utils import Candidate, CandidatePool


class PredictorNet(nn.Module):
    def __init__(self, architectural_input_size, training_recipe_input_size, embedding_size,
                 statistics_output_size, accuracy_output_size):
        super(PredictorNet, self).__init__()
        self.architectural_input_size = architectural_input_size
        self.training_recipe_input_size = training_recipe_input_size

        self.architecture_encoder = Architecture_Encoder(architectural_input_size, embedding_size)
        self.proxy_predictor = Proxy_Predictor(embedding_size, statistics_output_size)
        self.accuracy_predictor = Accuracy_Predictor(embedding_size + training_recipe_input_size,
                                                     accuracy_output_size)

    def forward(self, x, mode='proxy'):
        embedded_arch = self.architecture_encoder(x[:, :self.architectural_input_size, ...])
        if mode == 'accuracy':
            return self.accuracy_predictor(torch.cat([embedded_arch, x[:, self.architectural_input_size:, ...]], dim=1))
        elif mode == 'proxy':
            return self.proxy_predictor(embedded_arch)
        else:
            raise KeyError('mode should be \'accuracy\' or \'proxy\'')


class Architecture_Encoder(nn.Module):
    def __init__(self, architectural_input_size, embedding_size):
        super(Architecture_Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=architectural_input_size, out_features=256, bias=True)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=128, out_features=embedding_size, bias=True)

    def forward(self, arch):
        x = self.dp1(F.relu(self.fc1(arch)))
        x = self.dp2(F.relu(self.fc2(x)))
        return self.fc3(x)


class Proxy_Predictor(nn.Module):
    def __init__(self, embedding_size, statistics_output_size):
        super(Proxy_Predictor, self).__init__()
        self.dp = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=embedding_size, out_features=statistics_output_size, bias=True)

    def forward(self, embedded_arch):
        return self.fc(self.dp(F.relu(embedded_arch)))


class Accuracy_Predictor(nn.Module):
    def __init__(self, concat_input_size, accuracy_output_size):
        super(Accuracy_Predictor, self).__init__()
        self.fc1 = nn.Linear(in_features=concat_input_size, out_features=256, bias=True)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=128, out_features=accuracy_output_size, bias=True)

    def forward(self, concat_input):
        x = self.dp1(F.relu(self.fc1(concat_input)))
        x = self.dp2(F.relu(self.fc2(x)))
        return self.fc3(x)


class SampleDataset(Dataset):
    def __init__(self, data: Tensor, labels: Tensor):
        super(SampleDataset, self).__init__()
        assert len(data) == len(labels), '{} {}'.format(len(data), len(labels))
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx], self.labels[idx]


class Predictor():
    def __init__(self, architectural_input_size, training_recipe_input_size, embedding_size=24,
                 statistics_output_size=2, accuracy_output_size=1, net_class=PredictorNet):
        self.net = net_class(architectural_input_size, training_recipe_input_size, embedding_size,
                             statistics_output_size, accuracy_output_size)

    def predict(self, candidate: Candidate):
        with torch.no_grad():
            self.net.eval()
            candidate.predict_acc = [self.net(torch.tensor([candidate.e_data]), mode='accuracy').item()]
            self.net.train()

    def predict_all(self, candidates: CandidatePool):
        with torch.no_grad():
            self.net.eval()
            for candidate in candidates:
                candidate.predict_acc = [self.net(torch.tensor([candidate.e_data]), mode='accuracy').item()]
            self.net.train()

    def train_predictor(self, candidates: CandidatePool, mode='proxy',
                        dataset_split_rate=0.8, batch_size=100, epoch=10):
        assert mode == 'proxy' or mode == 'accuracy'
        data, labels = list(), list()
        split = int(len(candidates) * dataset_split_rate)
        for candidate in candidates:
            data.append(candidate.e_data)
            if mode == 'proxy':
                labels.append(candidate.statistics_label)
            elif mode == 'accuracy':
                labels.append(candidate.acc_label)
        train_dataset = SampleDataset(torch.tensor(data[:split]), torch.tensor(labels[:split]))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if split < len(candidates):
            test_dataset = SampleDataset(torch.tensor(data[split:]), torch.tensor(labels[split:]))
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # hard code
        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        print('[{} predictor] Begin training ...'.format(mode), flush=True)
        for i in range(epoch):
            train_loss, test_loss = 0, 0
            self.net.train()
            for data, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = self.net(data, mode=mode)
                loss = criterion(outputs, labels)
                train_loss += loss
                loss.backward()
                optimizer.step()
            train_loss = train_loss / split
            if split < len(candidates):
                with torch.no_grad():
                    self.net.eval()
                    for data, labels in test_dataloader:
                        outputs = self.net(data, mode=mode)
                        loss = criterion(outputs, labels)
                        test_loss += loss
                test_loss = test_loss / (len(candidates) - split)
            print('[{} predictor] [Epoch {:3d}] train loss: {} test loss: {}'.format(mode, i, train_loss, test_loss), flush=True)
        print('[{} predictor] End training ...'.format(mode), flush=True)
