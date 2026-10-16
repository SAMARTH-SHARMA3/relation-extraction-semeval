# Relation Extraction with SemEval-2010

**Comparing an SVM with a multi-head attention BiLSTM for classifying the semantic relationship between pairs of entities in a sentence.**

A group project by Richard Hull, Samarth Sharma, Maria Niculae and Akash Memon · 2025

---

## Materials

| | |
|:--|:--|
| **Project report** | Available on request |

To request it, [message me on LinkedIn](https://www.linkedin.com/in/samarth-sharma-120858242/) or [email me](https://mail.google.com/mail/?view=cm&to=samarth841sharma@gmail.com&su=Relation%20extraction%20report%20request).

---

## The task

Relation extraction is the problem of working out how two entities in a text are connected. Given a sentence with two marked entities, the model must name the relation between them and its direction.

> The `<e1>fire</e1>` inside WTC was caused by exploding `<e2>fuel</e2>.`
> **Relation:** Cause-Effect(e2, e1). The fuel is the cause and the fire is the effect.

## Dataset

The project uses the **SemEval-2010 Task 8** dataset. It has nine relation types: Cause-Effect, Instrument-Agency, Product-Producer, Content-Container, Entity-Origin, Entity-Destination, Component-Whole, Member-Collection and Message-Topic. Once direction is taken into account, this gives **19 classes**.

The classes are badly imbalanced. One label has only a single example in the whole dataset, which hurt model performance and shaped how we read the results.

## Approach

We built two models and measured what each enhancement added.

### SVM

A linear-kernel SVM trained on engineered features, using spaCy for tokenisation and dependency parsing, NLTK for preprocessing and POS tagging, Gensim for word vectors and scikit-learn for the pipeline. Feature sets were fused with `hstack`. The features were:

- **TF-IDF** with unigrams, bigrams and stop-word removal
- **Part-of-speech tags**
- **Dependency features**
- **GloVe-100 embeddings**
- **Entity position encoding**, which weights words by their distance from the two entities

An XGBoost model was trained as a comparison point.

### BiLSTM

A bidirectional LSTM in PyTorch with an embedding layer, a BiLSTM layer, a fully connected layer, softmax and dropout. We tested it in four stages: a base model, then GloVe embeddings, then a classic attention layer, then multi-head attention.

Hyperparameters were tuned on the 8,000-sample training set.

| Hyperparameter | Value |
|:--|:-:|
| Embedding size | 100 |
| Hidden state size | 50 |
| Layers | 1 |
| Learning rate | 0.005 |
| Weight decay | 0.0001 |
| Epochs | 50 |

Weight decay had the largest effect. All of the ten best-performing runs used 0.0001.

## Results

### BiLSTM

| Model | F1 |
|:--|:-:|
| Base BiLSTM | 56.7 |
| + GloVe embeddings | 58.7 |
| + Attention layer | 82.6 |
| **+ Multi-head attention** | **82.9** |
| *Zhou et al. (2016), BiLSTM* | *80.7* |
| *Zhou et al. (2016), attention BiLSTM* | *82.5* |

Pre-trained embeddings helped a little, but attention produced the big jump. Our multi-head attention model reached 82.9 F1, comparable to the 82.5 reported for the attention BiLSTM of Zhou et al. We suspect attention helps the model cope with the class imbalance by letting it focus on the most informative words.

### SVM

The best SVM reached **67% accuracy** and a **66% weighted F1**. Adding features one by one:

| Features | Accuracy (%) |
|:--|:-:|
| Dependency features alone | 24 |
| Word2Vec embeddings | 36 |
| GloVe + POS tags | 43 |
| GloVe embeddings | 51 |
| TF-IDF | 59 |
| TF-IDF + POS + dependency | 63 |
| TF-IDF + POS tags | 64 |
| **GloVe-100 + TF-IDF + POS + dependency** | **67** |

XGBoost reached 58% accuracy and 59% F1 on the same task.

## Key findings

- **Attention BiLSTM beat the SVM.** The multi-head attention BiLSTM clearly outperformed the best SVM, even though the SVM benefited from many hand-built linguistic features.
- **Combining features helped the SVM.** No single feature set was enough. Dense GloVe vectors alone scored 51%, and the best result came from fusing them with TF-IDF, POS and dependency features.
- **Imbalance matters.** Both models struggled on rare classes.

## Limitations

- **SVM:** It relies heavily on feature engineering, has trouble with context and is sensitive to class imbalance.
- **BiLSTM:** It inherits bias from the training data, takes a long time to train and needs a lot of compute to tune.
- **Next steps:** The report points to transformer-based models as a natural next step for this task.

## Tech

| | |
|:--|:--|
| Language | Python |
| Deep learning | PyTorch |
| Classical ML | scikit-learn, XGBoost |
| NLP | spaCy, NLTK, Gensim |
| Embeddings | GloVe |

`Python` · `PyTorch` · `scikit-learn` · `XGBoost` · `spaCy` · `NLTK` · `Gensim` · `GloVe`

## Notes

The SemEval-2010 Task 8 dataset is public and contains no personal or sensitive information. See Hendrickx et al. (2010).
