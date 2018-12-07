from PIL import Image
img_path='/Users/wywy/Desktop/test.jpg'

img=Image.open(img_path)
out=img.resize((512,512),Image.ANTIALIAS)
out.save('/Users/wywy/Desktop/test1.jpg')