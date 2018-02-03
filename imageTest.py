#!/usr/bin/python3

from PIL import Image
import numpy as np
import os

from network import Model


def load_images(directory):
    result = np.array([], dtype=np.uint8).reshape(0, Model.size, Model.size, 3)
    names = os.listdir(directory)
    names = [x for x in names if not x.startswith('.')]
    names = sorted(names)
    for name in names:
        image = Image.open(directory + '/' + name)
        image = image.convert('RGB')
        image = image.resize((Model.size, Model.size))
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)
    return result

model = Model()

training = load_images('training')
testing = load_images('testing')

j = 0
while True:
    loss = model.train(training, training)
    model.save()
    print(loss)

    # for i in range(testing.shape[0]):
    #     pixels = testing[i:i + 1:1][0]
    #     result = model.run([pixels])[0]
    #     pixels = np.concatenate([pixels, result], axis=1)
    #     image = Image.fromarray(pixels, "RGB")
    #     image = image.resize((512, 256))
    #     image.save('results/image' + str(i) + '.png')

    pixels = testing[0:1:1][0]
    result = model.run([pixels])[0]
    diff = np.absolute(pixels - result)
    pixels = np.concatenate([pixels, result, diff], axis=1)
    image = Image.fromarray(pixels, "RGB")
    image = image.resize((256 * 3, 256))
    image.save('results/image' + str(j).zfill(3) + '.png')

    j += 1
