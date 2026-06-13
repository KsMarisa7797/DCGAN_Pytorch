from PIL import Image
import glob

imgs_path = glob.glob('./data/*.png')
for path in imgs_path[:10]:
    img = Image.open(path)
    img.show()