from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_shape = (3,), activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model.summary()

model2 = Sequential()
model2.add(Dense(8, input_shape = (3,), activation = "tanh"))
model2.add(Dense(1, activation = "sigmoid"))
model2.summary()

model3 = Sequential()
model3.add(Dense(8, input_shape = (3,), activation = "sigmoid"))
model3.add(Dense(1, activation = "sigmoid"))
model3.summary()