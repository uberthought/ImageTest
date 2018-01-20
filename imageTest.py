#!/usr/bin/python3

from PIL import Image
import numpy as np

image = Image.open('test.jpg')

size = image.size

image = image.convert('RGB')
image = image.resize((256, 256))

pixels = np.array(image)

red = pixels[:, :, (0,)]
green = pixels[:, :, (1,)]
blue = pixels[:, :, (2,)]

zeros = np.zeros(image.size + (1,), dtype=pixels.dtype)

pixels2 = np.concatenate((red, green, zeros), axis=2)

# pixels2 = pixels2[0::4,0::4]

print(pixels2.shape)

image2 = Image.fromarray(pixels2, "RGB")
size = [int(x/4) for x in size]
image2 = image2.resize(size)

image2.save('test2.jpg')