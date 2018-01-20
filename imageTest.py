#!/usr/bin/python3

from PIL import Image
import numpy as np

from network import Model

model = Model()

image = Image.open('test.jpg')

size = image.size

image = image.convert('RGB')
image = image.resize((Model.size, Model.size))
pixels = np.array(image)

loss = model.train([pixels], [pixels])
model.save()
print(loss)

pixels2 = model.run([pixels])[0]
image2 = Image.fromarray(pixels2, "RGB")
image2 = image2.resize(size)

image2.save('test2.jpg')