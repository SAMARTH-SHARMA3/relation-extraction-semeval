import numpy as np


def load_glove_embeddings(embedding_size, vocab_df):
  glove = open("glove.6B.300d.txt", 'r', encoding="utf8")
  embeddings_dictionary = dict()

  for line in glove:
    values = line.split()
    word = values[0]
    vector_dimensions = np.asarray(values[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
  glove.close()

  emb_matrix = np.zeros((len(vocab_df.get_str_2_idx()), embedding_size))
  for word, index in vocab_df.get_str_2_idx().items():
    vect = embeddings_dictionary.get(word)
    if vect is None:
      emb_matrix[index] = np.random.randn(embedding_size)
    else:
      emb_matrix[index] = vect[:embedding_size]
  return emb_matrix

