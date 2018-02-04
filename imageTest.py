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

        # original
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

        # 90
        image = image.rotate(90)
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

        # 180
        image = image.rotate(90)
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

        # 270
        image = image.rotate(90)
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

        # flip
        image = image.rotate(90)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        pixels = np.array(image)

        # flip 90
        result = np.concatenate((result, [pixels]), axis=0)
        image = image.rotate(90)
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

        # flip 180
        image = image.rotate(90)
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

        # flip 270
        image = image.rotate(90)
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)
    return result

model = Model()

training = load_images('training')
testing = load_images('testing')

if not os.path.exists('results'):
    os.makedirs('results')

j = 0
while True:
    loss = model.train(training, training)
    model.save()
    print(loss)

    for i in range(testing.shape[0]):
        pixels = testing[i:i + 1:1][0]
        result = model.run([pixels])[0]
        pixels = np.concatenate([pixels, result], axis=1)
        image = Image.fromarray(pixels, "RGB")
        image = image.resize((256 * 2, 256))
        image.save('results/image' + str(i).zfill(3) + '.png')

    # pixels = testing[0:1:1][0]
    # result = model.run([pixels])[0]
    # diff = np.absolute(pixels - result)
    # diff = (diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]) / 3
    # diff = np.dstack([diff, diff, diff])
    # diff = diff.astype('uint8')
    # pixels = np.concatenate([pixels, result, diff], axis=1)
    # image = Image.fromarray(pixels, "RGB")
    # image = image.resize((256 * 3, 256))
    # image.save('results/image' + str(j).zfill(3) + '.png')

    j += 1
