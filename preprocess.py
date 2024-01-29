import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def crop_to_square_and_resize(img, pixel):
    h, w = img.shape[:2]
    square_size = min(img[:,:,0].shape)
    x = w//2 - square_size//2
    y = h//2 - square_size//2
    return cv.resize(img[y:y+square_size, x:x+square_size], (pixel, pixel),\
         interpolation=cv.INTER_LANCZOS4)

def equalize(img):
    B, G, R = cv.split(img)
    clahe = cv.createCLAHE(3)
    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)
    clahe = cv.merge((B, G, R))
    return clahe

def equalize_gray(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(2)
    clahe = clahe.apply(gray_img)
    return clahe

def color_correct(image, kernel_size):
    blurred = cv.blur(image, ksize=(kernel_size, kernel_size))
    result = cv.addWeighted(image, 4, blurred, -4, 128)
    return result

def extract_bv(image):
	b,green_fundus,r = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)

	# applying alternate sequential filtering (3 times closing opening)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)

	# removing very small contours through area parameter noise removal
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255
	contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

	# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
	#vessels and also in an interval of area
	fundus_eroded = cv2.bitwise_not(newfin)
	xmask = np.ones(green_fundus.shape[:2], dtype="uint8") * 255
	xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
			shape = "circle"
		else:
			shape = "veins"
		if(shape=="circle"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)

	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
	blood_vessels = cv2.bitwise_not(finimage)
	return finimage

if __name__ == 'main':
    test_image = cv.imread("images/15_left.jpg")
    test_image = crop_to_square_and_resize(test_image, 1024)
    img = extract_bv(test_image)
    extracted = cv.bitwise_and(test_image, test_image, mask = img)
    plt.figure(figsize=(15,15))
    plt.imshow(cv.cvtColor(test_image, cv.COLOR_BGR2RGB))
