# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:33:53 2018

@author: joshua
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

fig1 = plt.figure(figsize=(10,10),dpi=200) #create plot for histogram equalization results

img1 = cv2.imread('Lena.bmp',0) #read original images
img2 = cv2.imread('Cameraman.bmp',0)
img3 = cv2.imread('Peppers.bmp',0)

fig1.add_subplot(6, 4 , 1) #add original images to plot
plt.title('Lena')
plt.imshow(img1,cmap = 'gray')
fig1.add_subplot(6, 4 , 9)
plt.title('Cameraman')
plt.imshow(img2,cmap = 'gray')
fig1.add_subplot(6, 4 , 17)
plt.title('Peppers')
plt.imshow(img3,cmap = 'gray')

fig1.add_subplot(6, 4 , 5) # add original image's histogram to plot
plt.hist(img1.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 13)
plt.hist(img2.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 21)
plt.hist(img3.ravel(),256,[0,256])

"""
Part1 - Histogram Equalization
"""

equalizedImg1 = cv2.equalizeHist(img1) #add histogram equalized image to plot
fig1.add_subplot(6, 4 , 2)
plt.title('Histogram Equalization')
plt.imshow(equalizedImg1,cmap = 'gray')
equalizedImg2 = cv2.equalizeHist(img2)
fig1.add_subplot(6, 4 , 10)
plt.title('Histogram Equalization')
plt.imshow(equalizedImg2,cmap = 'gray')
equalizedImg3 = cv2.equalizeHist(img3)
fig1.add_subplot(6, 4 , 18)
plt.title('Histogram Equalization')
plt.imshow(equalizedImg3,cmap = 'gray')

fig1.add_subplot(6, 4, 6) # add histogram equalized image's histogram to plot
plt.hist(equalizedImg1.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 14)
plt.hist(equalizedImg2.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 22)
plt.hist(equalizedImg3.ravel(),256,[0,256])

"""
Part2 - Gamma Transformation
"""
def adjust_gamma(image, gamma=1.0):  #Output = Input^gamma
	table = np.array([((i / 255.0) ** gamma) * 255 #build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table) # apply gamma correction using the lookup table


gammaCorrectedImg1 = adjust_gamma(img1, 0.3) #perform gamma correction with gamma value 0.3, 0.5, 5 and add image to plot
fig1.add_subplot(6, 4 , 3)
plt.title('Gamma Correction 0.3')
plt.imshow(gammaCorrectedImg1 ,cmap = 'gray')
gammaCorrectedImg2 = adjust_gamma(img2, 0.5)
fig1.add_subplot(6, 4 , 11)
plt.title('Gamma Correction 0.5')
plt.imshow(gammaCorrectedImg2 ,cmap = 'gray')
gammaCorrectedImg3 = adjust_gamma(img3, 5)
fig1.add_subplot(6, 4 , 19)
plt.title('Gamma Correction 0.5')
plt.imshow(gammaCorrectedImg3 ,cmap = 'gray')

fig1.add_subplot(6, 4 , 7) # add gamma corrected image's histogram to plot
plt.hist(gammaCorrectedImg1.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 15)
plt.hist(gammaCorrectedImg2.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 23)
plt.hist(gammaCorrectedImg3.ravel(),256,[0,256])

"""
Part3 - Image Sharpening using the Laplacian
"""
def normalize(img):
    img = img - np.min(img)
    img = img * 255.0/np.max(img)
    return img

def laplacian_sharpen(img):
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    float64Img = img * 1.0
    lapImg = cv2.filter2D(float64Img,-1,kernel);
    normLap = normalize(lapImg)
    output = float64Img - normLap
    return output

sharpened1 = laplacian_sharpen(img1) # perform Laplacian Sharpen and add image to plot
fig1.add_subplot(6,4,4)
plt.title('Sharpening using Laplacian')
plt.imshow(sharpened1,cmap='gray')
sharpened2 = laplacian_sharpen(img2)
fig1.add_subplot(6,4,12)
plt.title('Sharpening using Laplacian')
plt.imshow(sharpened2,cmap='gray')
sharpened3 = laplacian_sharpen(img3)
fig1.add_subplot(6,4,20)
plt.title('Sharpening using Laplacian')
plt.imshow(sharpened3,cmap='gray')

fig1.add_subplot(6, 4 , 8) # add sharpened image's histogram to plot
plt.hist(sharpened1.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 16)
plt.hist(sharpened2.ravel(),256,[0,256])
fig1.add_subplot(6, 4 , 24)
plt.hist(sharpened3.ravel(),256,[0,256])


plt.tight_layout()
plt.show() #show final plot

"""
plt.hist(equalizedImg.ravel(),256,[0,256])

hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.hist(img.ravel(),256,[0,256])
plt.savefig('lenaHist.png');
plt.gcf().clear()


equalizedImg = cv2.equalizeHist(img)
plt.hist(equalizedImg.ravel(),256,[0,256])
plt.savefig('lenaEqualizedHist.png');

lenaHist = cv2.imread('lenaHist.png',0)
lenaEqualizedHist = cv2.imread('lenaEqualizedHist.png',0)

fig = plt.figure(figsize=(30,30))
temp = img 
fig.add_subplot(1, 4 , 1)
plt.imshow(temp,cmap = 'gray')

temp =  lenaHist
fig.add_subplot(1, 4 , 2)
plt.imshow(temp)

temp =  equalizedImg
fig.add_subplot(1, 4 , 3)
plt.imshow(temp,cmap = 'gray')

temp =  lenaEqualizedHist
fig.add_subplot(1, 4 , 4)
plt.imshow(temp)

res = np.hstack((img,equalizedImg))
hist = np.hstack((lenaHist,lenaEqualizedHist))
cv2.imshow('equalized',res)
"""