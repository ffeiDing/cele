from PIL import Image
import glob
image_list = []
for filename in glob.glob('/Users/dingfeifei/Desktop/cele/cele_images/training/*.jpg'):
    im = Image.open(filename)
    image_list.append(im)

print(image_list)
print(len(image_list))
image = image_list[0]