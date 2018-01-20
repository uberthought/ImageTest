#!/usr/bin/python3

from PIL import Image
import numpy as np

image = Image.open('test.jpg').convert('RGB')

pixels = np.array(image)

pixels = pixels[:, :, (1, 2)]

a = pixels.shape

zeros = np.zeros(pixels.shape[0:2]).reshape(pixels.shape[0], pixels.shape[1], 1)

print(zeros.shape)
print(pixels.shape)

foo = np.concatenate((pixels, zeros), axis=2)

print(foo.shape)

image2 = Image.fromarray(foo, "RGB")

image2.save('test2.jpg')