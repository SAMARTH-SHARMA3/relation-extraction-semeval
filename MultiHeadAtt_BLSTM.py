import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self):
        self.path_word = "glove.6B.100d.txt"
        self.word_dim = 100  # dimension of word embedding

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character
        word2id['UNK'] = len(word2id)  # out of vocabulary
        word2id['<e1>'] = len(word2id)
        word2id['<e2>'] = len(word2id)
        word2id['</e1>'] = len(word2id)
        word2id['</e2>'] = len(word2id)

        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        word_vec = np.stack(word_vec)
        vec_mean, vec_std = word_vec.mean(), word_vec.std()
        special_emb = np.random.normal(vec_mean, vec_std, (6, self.word_dim))
        special_emb[0] = 0  # <pad> is initialize as zero

        word_vec = np.concatenate((special_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec

class RelationLoader(object):
    def __load_relation(self):
        rel2id = {}
        id2rel = {}
        with open('relation2id.txt', 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()

class SemEvalDateset(Dataset):
    def __init__(self, filename, rel2id, word2id):
        self.filename = filename
        self.rel2id = rel2id
        self.word2id = word2id
        self.max_len = 100
        self.dataset, self.label = self.__load_data()

    def __symbolize_sentence(self, sentence):
        """
            Args:
                sentence (list)
        """
        mask = [1] * len(sentence)
        words = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

        unit = np.asarray([words, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 2, self.max_len))
        return unit

    def __load_data(self):
        data = []
        labels = []
        with open(self.filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                label_idx = self.rel2id[label]

                one_sentence = self.__symbolize_sentence(sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

class SemEvalDataLoader(object):
    def __init__(self, rel2id, word2id):
        self.rel2id = rel2id
        self.word2id = word2id

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def __get_data(self, filename, shuffle):
        dataset = SemEvalDateset(filename, self.rel2id, self.word2id)
        loader = DataLoader(
            dataset=dataset,
            batch_size=10,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        return self.__get_data('train.json', shuffle=True)

    def get_test(self):
        return self.__get_data('test.json', shuffle=False)

class MultiHeadAtt_BLSTM(nn.Module):
    def __init__(self, word_vec, class_num):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = 100
        self.word_dim = 100
        self.hidden_size = 100
        self.layers_num = 1
        self.emb_dropout_value = 0.3
        self.num_heads = 4
        self.lstm_dropout_value = 0.3
        self.linear_dropout_value = 0.5

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True
        )

        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        # initialize weight
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        h = torch.sum(h, dim=2)  # B*L*H
        return h

    def attention_layer(self, h, mask):
        # Compute self-attention
        attn_output, _ = self.multihead_attn(h, h, h, key_padding_mask=mask.eq(0))

        # Pooling: Take mean across sequence length
        reps = attn_output.mean(dim=1)  # B*H
        reps = F.tanh(reps)
        return reps

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        mask = data[:, 1, :].view(-1, self.max_len)
        emb = self.word_embedding(token)  # B*L*word_dim
        emb = self.emb_dropout(emb)
        h = self.lstm_layer(emb, mask)  # B*L*H
        h = self.lstm_dropout(h)
        reps = self.attention_layer(h, mask)  # B*reps
        reps = self.linear_dropout(reps)
        logits = self.dense(reps)
        return logits


def semeval_scorer(predict_label, true_label, class_num=10):
    import math
    assert true_label.shape[0] == predict_label.shape[0]
    confusion_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float32)
    xDIRx = np.zeros(shape=[class_num], dtype=np.float32)
    for i in range(true_label.shape[0]):
        true_idx = math.ceil(true_label[i] / 2)
        predict_idx = math.ceil(predict_label[i] / 2)
        if true_label[i] == predict_label[i]:
            confusion_matrix[predict_idx][true_idx] += 1
        else:
            if true_idx == predict_idx:
                xDIRx[predict_idx] += 1
            else:
                confusion_matrix[predict_idx][true_idx] += 1

    col_sum = np.sum(confusion_matrix, axis=0).reshape(-1)
    row_sum = np.sum(confusion_matrix, axis=1).reshape(-1)
    f1 = np.zeros(shape=[class_num], dtype=np.float32)

    for i in range(0, class_num):  # ignore the 'Other'
        try:
            p = float(confusion_matrix[i][i]) / float(col_sum[i] + xDIRx[i])
            r = float(confusion_matrix[i][i]) / float(row_sum[i] + xDIRx[i])
            f1[i] = (2 * p * r / (p + r))
        except:
            pass
    actual_class = 0
    total_f1 = 0.0
    for i in range(1, class_num):
        if f1[i] > 0.0:  # classes that not in the predict label are not considered
            actual_class += 1
            total_f1 += f1[i]
    try:
        macro_f1 = total_f1 / actual_class
    except:
        macro_f1 = 0.0
    return macro_f1


class Eval(object):
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, criterion, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0
        with torch.no_grad():
            model.eval()
            for _, (data, label) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                logits = model(data)
                loss = criterion(logits, label)
                total_loss += loss.item() * logits.shape[0]

                _, pred = torch.max(logits, dim=1)  # replace softmax with max function, same impacts
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))
                predict_label.append(pred)
                true_label.append(label)
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        eval_loss = total_loss / predict_label.shape[0]

        f1 = semeval_scorer(predict_label, true_label)
        return f1, eval_loss, predict_label


def train(model, criterion, loader, device):
    num_epoch = 30

    train_loader, test_loader = loader
    optimizer = optim.Adadelta(
        model.parameters(), lr=1.0, weight_decay=1e-5)

    print(model)
    print('Training model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('Starting to train the model ...')

    eval_tool = Eval(device)
    max_f1 = -float('inf')
    for epoch in range(1, num_epoch + 1):
        for data, labels in train_loader:
            model.train()
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
            optimizer.step()

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, test_loss, _ = eval_tool.evaluate(model, criterion, test_loader)

        print('[%03d/%03d] train_loss: %.3f | test_loss: %.3f | micro f1 on dev: %.4f'
              % (epoch, num_epoch, train_loss, test_loss, f1), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), f'model_multiheadatt.pkl')
            print('>>> saved model!')
        else:
            print()

def preprocess_sentence(sentence, word2id, max_len=100):
    mask = [1] * len(sentence)
    words = []

    length = min(max_len, len(sentence))
    mask = mask[:length]

    for i in range(length):
        words.append(word2id.get(sentence[i].lower(), word2id['UNK']))

    if length < max_len:
        mask.extend([0] * (max_len - length))  # 'PAD' mask is zero
        words.extend([word2id['PAD']] * (max_len - length))

    unit = np.asarray([words, mask], dtype=np.int64)
    unit = np.reshape(unit, newshape=(1, 2, max_len))

    return torch.tensor(unit)  # Return a torch tensor for model input

def predict_label(model, input):
    batch_outputs = model(input)
    y_predictions = np.argmax(batch_outputs.cpu().detach().numpy(), axis=1)
    return y_predictions