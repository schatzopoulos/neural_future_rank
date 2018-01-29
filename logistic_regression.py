import numpy as np
from keras.models import Sequential
from keras.layers import Dense


x_train = np.matrix([
	[0, 1], 
	[1, 0], 
	[1, 0], 
	[1, 0], 
	[0, 1], 
	[1, 0], 
	[0, 1], 
	[0, 1]
])
y_train = np.matrix([[
	1, 0, 0, 0, 1, 0, 1, 1
]])

x_test = np.matrix([
	[1, 0], 
	[1, 0], 
	[0, 1]
])
y_test = np.matrix([[
	0, 0, 1
]])

print x_train.shape
model = Sequential()
out_dim = 1
input_dim = x_train.shape[1]
model.add(Dense(out_dim, activation='sigmoid', input_shape=input_dim))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])