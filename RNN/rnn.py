""" Modules Required are:

1. tensorflow
2. tensorflow_datasets
3. matplotlib

pip install module-name
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# create a helper function to plot graphs:
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

# Load Dataset for IMDB movie review 
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# The dataset info includes the encoder (a tfds.features.text.SubwordTextEncoder).
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

# This text encoder will reversibly encode any string, falling back to byte-encoding if necessary.
sample_string = 'Hello TensorFlow.'
encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))
original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

assert original_string == sample_string
