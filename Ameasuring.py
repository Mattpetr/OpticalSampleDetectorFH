import cv2
import numpy as np
from basicFunctions import*


img = cv2.imread("Assets/3/140322_3.png")

#cv2.imshow("Orig",img)
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
cv2.imshow("Resized",img)

# if selectingROI == True:
#     r = cv2.selectROI(img)
#     img = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
#     print(f"{int(r[1])}:{int(r[1] + r[3])},{int(r[0])}:{int(r[0] + r[2])}")
# img = img[5*344:5*381,5*300:5*327]
# cv2.imwrite("colour1.png",img)
# copy = img.copy()

#binaer,colour = color_filter(img, [80, 0, 0], [120, 255, 255], True)   #<--- hat gut geklappt
#binaer,colour = color_filter(img, [10, 0, 0], [178, 255, 255], True)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# plt.title('Hue')
# plt.xlabel('Bins')
# plt.ylabel('sum of pixels')
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()

#ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

cv2.imshow("thresg",thresh)

erode = cv2.erode(thresh,(3,3), iterations=200)
cv2.imshow("Erode",erode)
dilate = cv2.dilate(erode,(3,3),iterations=100)
cv2.imshow("Dilate",dilate)

merged = cv2.bitwise_xor(thresh,dilate,mask=None)
cv2.imshow("Merged",merged)
# img = cv2.bitwise_and(gray,gray,mask=thresh)
#
# cv2.imshow("mer",img)
# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image=gray, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cv2.imshow("contours",img)
# edges = cv2.Canny(thresh,30,100)

# #histo_HSV(img,"h",size_factor=2)
# cv2.imshow("bins",edges)
cv2.waitKey(0)

