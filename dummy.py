import os
import numpy as np
import Image

images = os.listdir('./train')
img = os.path.join(images[1])
imgs = np.array(Image.open(img))
