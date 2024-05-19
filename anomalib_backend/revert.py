from PIL import Image  
  
# 打开图片  
img = Image.open('TrainSet/abnormal/flatness/1黑色小产品-B模-20240120-094947--原图NG.png')  
print(img.size)
# 上下翻转  
#img_inverted = img.transpose(Image.FLIP_TOP_BOTTOM)  
  
# 保存翻转后的图片  
#img_inverted.save('TrainSet/abnormal/flatness/1黑色小产品-B模-20240120-094947--原图NG1.png')