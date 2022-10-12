import tensorflow as tf
import pandas as pd
import numpy as np

path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
df = pd.read_csv(path)

train_percentage = 0.66
train_last_num = np.floor(df.last_valid_index()*train_percentage).astype(int)
train_df = df[0:train_last_num]
test_df = df[train_last_num:-1]

# train_df = train_df.reindex(np.random.permutation(train_df))

train_x, train_y = train_df.values[:, :-1], train_df.values[:, -1]
train_x= ((train_x - train_x.mean())/train_x.std()).astype('float32')

#if class is g set to true
train_y = (train_y == 'g').astype(float)

test_x, test_y = test_df.values[:, :-1], test_df.values[:, -1]
test_x= ((test_x - test_x.mean())/test_x.std()).astype('float32')

#if class is g set to true
test_y = (test_y == 'g').astype(float)


input_shape = train_x.shape[1]

model = tf.keras.models.Sequential([])
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(input_shape,)))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=150, batch_size=32)

print("test loss")
loss, acc = model.evaluate(test_x, test_y)