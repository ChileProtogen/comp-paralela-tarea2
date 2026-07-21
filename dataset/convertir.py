from PIL import Image
import os
for f in os.listdir('.'):
    if f.endswith('.png'):
        Image.open(f).save(f.replace('.png', '.bmp'), 'BMP')