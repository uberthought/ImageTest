#!/usr/bin/python3

from PIL import Image
import numpy as np
import os
import random

from network import Model

def load_test_images(directory):
    result = np.array([], dtype=np.uint8).reshape(0, Model.size, Model.size, 3)

    names = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            names += [os.path.join(root, name)]
    names = [x for x in names if not x.startswith('.')]
    
    names = sorted(names)

    for name in names:
        image = Image.open(name)
        image = image.convert('RGB')
        image = image.resize((Model.size, Model.size))
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

    return result

def load_train_names(directory):
    names = []

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            names += [os.path.join(root, name)]
    names = [x for x in names if not x.startswith('.')]
    
    return names

def load_images(names):
    result = np.array([], dtype=np.uint8).reshape(0, Model.size, Model.size, 3)

    for name in names:
        image = Image.open(name)
        image = image.convert('RGB')
        image = image.resize((Model.size, Model.size))

        # original
        pixels = np.array(image)
        result = np.concatenate((result, [pixels]), axis=0)

    return result

model = Model()

if not os.path.exists('results'):
    os.makedirs('results')

training_names = load_train_names('training')
testing = load_test_images('testing')

for j in range(10):
    training = load_images(random.sample(training_names, 4))
    loss = model.train(training, testing)

    for i in range(testing.shape[0]):
        pixels = testing[i:i + 1:1][0]
        result = model.run([pixels])[0]
        diff = np.absolute(pixels.astype('float') - result.astype('float'))
        diff = diff.astype('uint8')
        pixels = np.concatenate([pixels, result, diff], axis=1)
        image = Image.fromarray(pixels, "RGB")
        image = image.resize((256 * 3, 256))
        image.save('results/image' + str(i).zfill(3) + '.png')
