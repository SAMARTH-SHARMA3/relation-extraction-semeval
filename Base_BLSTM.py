import string

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class RelationClassification_Model(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers,class_num,pad_idx):
        super(RelationClassification_Model, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size,padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,bidirectional = True)
        self.linear = nn.Linear(2*hidden_size, class_num)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.7)

    def forward(self, X):
        embeddings = self.dropout(self.embedding(X))
        hiddens, _ = self.lstm(embeddings)
        x = hiddens.view((hiddens.shape[0], hiddens.shape[1], 2, self.hidden_size))
        concats = torch.cat((x[-1,:,0,:],x[0,:,1,:]), dim = 1)
        outputs = self.softmax(self.linear(concats))

        return outputs

def train_model(model, optimiser, criterion, epochs, training_generator, device):
  train_losses = []

  train_accs = []
  # Loop over epochs
  for epoch in tqdm(range(epochs)):
      running_trainloss, running_loss, train_acc = 0.0, 0.0, 0.0
      train_cnt, val_cnt = 0, 0

      # Training
      for batch_train_data, batch_train_labels in training_generator:
          # Transfer to GPU
          batch_train_data, batch_train_labels = batch_train_data.to(device), batch_train_labels.to(device)

          ### Model computations

          # Forward pass: compute predicted y by passing x to the model.
          batch_train_outputs = model(batch_train_data)

          # Compute loss.
          loss = criterion(batch_train_outputs, batch_train_labels)

          # Step weights
          optimiser.zero_grad()
          # Backward pass: compute gradient of the loss with respect to model parameters
          loss.backward()

          optimiser.step()

          # print statistics
          running_trainloss += loss.item()
          running_loss += loss.item()

          train_acc += torch.sum(torch.argmax(batch_train_outputs, dim =1) ==  batch_train_labels.long()) / torch.numel(batch_train_labels)

          train_cnt += 1

      train_losses.append(running_trainloss / train_cnt)
      train_accs.append(train_acc / train_cnt * 100)

      #scheduler.step()
  return [train_losses, train_accs]


def test_model(test_df, partition, vocab_df, device, model, batch_size):
    # Training and validation Data Generators
    test_set = SemEval2010_task8_Dataset(test_df, partition['test'], vocab_df)

    pad_idx = vocab_df.str_2_idx[""]

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0,
              'collate_fn': batch_collate_fn(pad_idx=pad_idx)
              }

    test_generator = torch.utils.data.DataLoader(test_set, **params)

    with torch.set_grad_enabled(False):
        for batch_data, batch_labels in test_generator:
            # Transfer to GPU
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Model computations
            # forward + backward + optimize
            batch_outputs = model(batch_data)

            y_predictions = np.argmax(batch_outputs.cpu().numpy(), axis=1)
            y_test = batch_labels.cpu().numpy()

            precision_metric = precision_score(y_test, y_predictions, average="macro")

            recall_metric = recall_score(y_test, y_predictions, average="macro")

            accuracy_metric = accuracy_score(y_test, y_predictions)

            f1_metric = f1_score(y_test, y_predictions, average="macro")

            return [precision_metric, recall_metric, accuracy_metric, f1_metric, y_predictions]


def get_relation_from_id(relation_id):
    match relation_id:
        case 0: return "Cause-Effect(e1,e2)"
        case 1: return "Cause-Effect(e2,e1)"
        case 2: return "Component-Whole(e1,e2)"
        case 3: return "Component-Whole(e2,e1)"
        case 4: return "Content-Container(e1,e2)"
        case 5: return "Content-Container(e2,e1)"
        case 6: return "Entity-Destination(e1,e2)"
        case 7: return "Entity-Destination(e2,e1)"
        case 8: return "Entity-Origin(e1,e2)"
        case 9: return "Entity-Origin(e2,e1)"
        case 10: return "Instrument-Agency(e1,e2)"
        case 11: return "Instrument-Agency(e2,e1)"
        case 12: return "Member-Collection(e1,e2)"
        case 13: return "Member-Collection(e2,e1)"
        case 14: return "Message-Topic(e1,e2)"
        case 15: return "Message-Topic(e2,e1)"
        case 16: return "Product-Producer(e1,e2)"
        case 17: return "Product-Producer(e2,e1)"
        case 18: return "Other"
    return "ERROR - relation_id NOT VALID!!"


class Vocabulary_and_Dataframe:
    def __init__(self, freq_threshold = 5, pretrained_vocab = None):
        self.df = None
        self.freq_threshold = freq_threshold

        if pretrained_vocab is None :
            self.frequencies = {}
            self.idx_2_str = {0: "", 1: ""}
            self.str_2_idx = {"": 0, "": 1}

        else :
            self.idx_2_str = pretrained_vocab.itos
            self.str_2_idx = pretrained_vocab.stoi
            self.idx_2_str.append("")
            self.str_2_idx[""] = len(self.str_2_idx)
            self.idx_2_str.append("")
            self.str_2_idx[""] = len(self.str_2_idx)
            self.idx_2_str.append("e1start")
            self.str_2_idx["e1start"] = len(self.str_2_idx)
            self.idx_2_str.append("e1end")
            self.str_2_idx["e1end"] = len(self.str_2_idx)
            self.idx_2_str.append("e2start")
            self.str_2_idx["e2start"] = len(self.str_2_idx)
            self.idx_2_str.append("e2end")
            self.str_2_idx["e2end"] = len(self.str_2_idx)


    def __len__(self):
        return len(self.idx_2_str)

    @staticmethod
    def preprocess(text):
        # tokenize
        text = text.split()
        # convert to lower case
        text = [word.lower().strip() for word in text]
        # remove punctuation from each token
        text = [w.translate(str.maketrans('', '', string.punctuation)) for w in text]

        return text

    def build_vocabulary_and_dataframe(self, input_df):

        self.df = input_df

        idx = 2
        i = 0
        for index, row in input_df.iterrows():
            sentence = row['sentence']
            relation_id = row['relation']

            sentence = sentence.replace("<e1>", " e1start ").replace("</e1>", " e1end ")
            sentence = sentence.replace("<e2>", " e2start ").replace("</e2>", " e2end ")

            words =  self.preprocess(sentence)
            for word in words:
                if word not in self.frequencies:
                    self.frequencies[word] = 1

                else:
                    self.frequencies[word] += 1

                if self.frequencies[word] == self.freq_threshold:
                    self.str_2_idx[word] = idx
                    self.idx_2_str[idx] = word
                    idx += 1

            self.df.loc[i] = [" ".join(words), relation_id]
            i+=1

    def IndexEncoding(self, text):
        tokenized_text = self.preprocess(text)
        index_encoded_text = [self.str_2_idx[token] if token in self.str_2_idx else self.str_2_idx[""]
                                for token in tokenized_text ]

        return index_encoded_text

    def get_str_2_idx(self):
        return self.str_2_idx

    def get_df(self):
        return self.df

class SemEval2010_task8_Dataset(torch.utils.data.Dataset):
    """
    Characterize SemEval2010_task8 Dataset for PyTorch. To be compatible with
    DataLoader and other PyTorch syntaxes.
    """
    def __init__(self, df, indices, vocab):

        self.df = df
        self.df = self.df.iloc[indices]

        # Get img names, caption columns
        self.sentences = self.df["sentence"].tolist()
        self.labels = self.df["relation"].tolist()

        self.vocab = vocab

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

    def __getitem__(self, index):
        """
        Generates one sample of data with label

        Arguments
        ------------
        index : integer, in range of len(self.list_IDs)
                index of sample in list of IDs,

        Returns
        ------------
        X : torch tensor,
            input data

        Y : torch tensor,
            correspond label of input data
        """
        # Select sample
        sentence = self.sentences[index]
        label = self.labels[index]

        tensor_sentence = torch.tensor(self.vocab.IndexEncoding(sentence))

        return tensor_sentence ,torch.tensor(label)

class batch_collate_fn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):

        sentences = [item[0] for item in batch]
        sentences = pad_sequence(sentences, batch_first=False, padding_value=self.pad_idx)

        labels = torch.tensor( [item[1] for item in batch])

        return sentences, labels