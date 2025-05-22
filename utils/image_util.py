import cv2
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt


def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) is str:
        img = cv2.imread(img)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


# Changed line 58 to 48 
# def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    
#     font = ImageFont.load_default()
    
#     if type(image) == np.ndarray:
#         color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(color_coverted)

#     if platform.system() == 'Darwin':
#         font = 'AppleGothic.ttf'
#     elif platform.system() == 'Windows':
#         font = 'malgun.ttf'

#     image_font = ImageFont.truetype(font, font_size)
    
#     draw = ImageDraw.Draw(image)

#     draw.text((x, y), text, font=image_font, fill=color)

#     numpy_image = np.array(image)
#     opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

#     return opencv_image

def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    
    if type(image) == np.ndarray:
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_converted)

    if platform.system() == 'Darwin':
        print("NO FONTS")
        font_path = '/Library/Fonts/AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    else:
        # If it's Linux or other systems, use a default font path
        # font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        font_path = '/usr/share/fonts/NanumFont/NanumGothicBold.ttf'

    image_font = ImageFont.truetype(font_path, font_size)
    
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image