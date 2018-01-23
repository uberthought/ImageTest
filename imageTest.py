#!/usr/bin/python3

from PIL import Image
import numpy as np
import os

from network import Model

model = Model()

names = os.listdir('images')

X = np.array([], dtype=np.float).reshape(0, Model.size, Model.size, 3)
Y = np.array([], dtype=np.float).reshape(0, Model.output_size)

i = 0
for name in names:
    image = Image.open('images/' + name)
    image = image.convert('RGB')
    image = image.resize((Model.size, Model.size))
    pixels = np.array(image)

    X = np.concatenate((X, [pixels]), axis=0)
    y = np.zeros(Model.output_size)
    y[i] = 1
    Y = np.concatenate((Y, [y]))

    i += 1

loss = model.train(X, Y)
model.save()

for i in range(X.shape[0]):
    pixels2 = X[i:i+1:1][0]
    result = model.run([pixels2])[0]
    print(i, np.argmax(result), result)
    image2 = Image.fromarray(pixels2.astype(np.int8), "RGB")
    image2.save('test' + str(i) + '.png')
    # image2 = image2.resize(size)
