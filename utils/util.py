import matplotlib.pyplot as plt
from PIL import Image

def visualize(meme_list, result):
  for i in result:
    path = meme_list[i]
    img = Image.open(path)

    plt.imshow(img)
    plt.axis('off')
    # plt.show()
    plt.savefig("image.png")