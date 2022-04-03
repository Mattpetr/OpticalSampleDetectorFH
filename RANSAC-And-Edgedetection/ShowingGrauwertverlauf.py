import cv2
from matplotlib import pyplot as plt
import numpy as np
from RANSACGitHub import *

#img = cv2.imread("../Assets/1/DSCN0001.JPG")
img = cv2.imread("../Assets/3/140322_3.png")

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
#img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
h,w = img.shape
factor = 1
hoehe = int(factor*h)
img = img[hoehe-h:hoehe,0:w]
cv2.imshow("cut",img)
cv2.waitKey(0)

img = cv2.resize(img,(0,0),fx=1,fy=1)
h,w = img.shape
print(w)


cv2.imshow("im",cv2.resize(img,(0,0),fx=5,fy=5))
print(img[0].shape)
xpoints = np.array([i for i in range(0,w)])
ypoints = np.array([i for i in img[0]])


#ransac(xpoints,ypoints)

plt.plot(xpoints, ypoints)
#plt.scatter(xpoints, ypoints, s=3)
plt.show()
cv2.waitKey(0)