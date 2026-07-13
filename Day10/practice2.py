from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential()

# input layer
model.add(Input(shape = (3,)))

# hidden layer
model.add(Dense(8, activation = "relu"))

# output layer
model.add(Dense(1, activation = "sigmoid"))

model.summary()