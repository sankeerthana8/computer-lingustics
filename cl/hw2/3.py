import numpy as np
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Concatenate, Reshape
from keras.models import Model

# Load data
def load_corpus(filename):
    data = pd.read_csv(filename, sep="\t")
    sentences = data['sentence'].tolist()
    labels = data['structure'].tolist()
    return sentences, labels

sentences, labels = load_corpus('training_sentences.txt')

# Split the data into train and test sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.1, random_state=42)

# Define symbolic features
symbolic_input = Input(shape=(10,), dtype='int32')
symbolic_embedding = Embedding(input_dim=1000, output_dim=32, input_length=10)(symbolic_input)
conv1d = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(symbolic_embedding)
pooling = MaxPooling1D(pool_size=2)(conv1d)
symbolic_output = Dense(1, activation='sigmoid')(pooling)

# Define embedding features
embedding_input = Input(shape=(5,), dtype='float32')
embedding_reshaped = Reshape(target_shape=(5, 1))(embedding_input)
embedding_conv1d = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(embedding_reshaped)
embedding_pooling = MaxPooling1D(pool_size=2)(embedding_conv1d)
embedding_output = Dense(1, activation='sigmoid')(embedding_pooling)

# Combine symbolic and embedding features
# Pad the second input tensor with zeros along the second dimension
padded_embedding_output = keras.layers.ZeroPadding1D(padding=(0, 3))(embedding_output)

# Concatenate the two tensors along the second dimension
concatenated = keras.layers.Concatenate(axis=1)([symbolic_output, padded_embedding_output])

#concatenated = Concatenate()([symbolic_output, embedding_output])
output = Dense(1, activation='sigmoid')(concatenated)

# Define the model
model = Model(inputs=[symbolic_input, embedding_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate some fake data
symbolic_data = np.random.randint(1000, size=(1000, 10))
embedding_data = np.random.randn(1000, 5)
labels = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit([symbolic_data, embedding_data], labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_symbolic_data = np.random.randint(1000, size=(len(test_sentences), 10))
test_embedding_data = np.random.randn(len(test_sentences), 5)
test_predictions = model.predict([test_symbolic_data, test_embedding_data])
test_predictions = np.round(test_predictions).flatten()
test_f1 = f1_score(test_labels, test_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f'Test F1 score: {test_f1}')
print(f'Test accuracy: {test_accuracy}')
