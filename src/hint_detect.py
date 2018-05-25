"""
For testing hint image processing purpose
"""
import cv2
import os
import matplotlib.pyplot as plt

def show(im):
    cv2.imshow("", im)
    cv2.waitKey(0)

def show_2(im1, im2):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)
    plt.show()
    
while 1:
    name = str(input("image: "))
    im = cv2.imread("images/{}.png".format(name))
    im = im[:,:,::-1]
    image = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    im1, contours, hierarchy = cv2.findContours(thresh1, 1, 2)
    print("{} contours found".format(len(contours)))
    result = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print("area: {}".format(w*h))
        if w*h >= 100:
            result.append((x + w/2)/image.shape[1])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(result)    
    show_2(image, thresh1)
    
