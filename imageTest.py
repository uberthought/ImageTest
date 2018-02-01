#!/usr/bin/python3

from PIL import Image
import numpy as np
import os

from network import Model

model = Model()

names = os.listdir('images')
names = [x for x in names if not x.startswith('.')]
names = sorted(names)

X = np.array([], dtype=np.uint8).reshape(0, Model.size, Model.size, 3)

for name in names:
    image = Image.open('images/' + name)
    image = image.convert('RGB')
    image = image.resize((Model.size, Model.size))
    pixels = np.array(image)

    X = np.concatenate((X, [pixels]), axis=0)

while True:
    loss = model.train(X, X)
    model.save()
    print(loss)

    for i in range(X.shape[0]):
        pixels = X[i:i+1:1][0]
        pixels = model.run([pixels])[0]
        image = Image.fromarray(pixels, "RGB")
        # image = image.resize((256, 256))
        image.save('results/test' + str(i) + '.png')
