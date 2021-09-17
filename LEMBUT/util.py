import numpy as np
from PIL import Image

def imgToMat(filepath: str):
  file = Image.open(filepath, "r")
  img = np.array(file)
  b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
  return b, g, r

# print("img")
# print(img)
# print("\nred channel\n")
# print(r)
# print("\ngreen channel\n")
# print(g)
# print("\nblue channel\n")
# print(b )