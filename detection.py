import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
from basicFunctions import*

selectingROI = False

img = cv2.imread("Assets/3/140322_9_edited.png")
size=0.2
img = cv2.resize(img,(0,0),fx=size,fy=size)


if selectingROI == True:
    r = cv2.selectROI(img)
    img = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    print(f"{int(r[1])}:{int(r[1] + r[3])},{int(r[0])}:{int(r[0] + r[2])}" "[y0:y1,x0:x1]")
# img = img[5*344:5*381,5*300:5*327]
# cv2.imwrite("colour1.png",img)
copy = img.copy()

#binaer,colour = color_filter(img, [94, 140, 0], [103, 255, 255], True)   #<--- hat gut geklappt
#binaer,colour = color_filter(img, [90, 0,0], [120, 180, 255], True)
#histo_HSV(colour,"s",size_factor=2)

cv2.imshow("orig",img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret3, binaer = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#ret, binaer = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

cv2.imshow("bins",binaer)
#cv2.waitKey(0)
biggest_contours=biggest_box(binaer,img=img,numBigCon=10)

stuetzpunkte = []
for i in biggest_contours:
    cv2.drawContours(image=img, contours=i[1], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    stuetzpunkte.append(i[1])



# pts = np.array(stuetzpunkte,np.int32)
# pts = pts.reshape((-1, 1, 2))

# copy = cv2.polylines(img=copy, pts=[pts], isClosed=True, color= (0,255,255), thickness= 1)
# copy = cv2.resize(copy,(0,0),fx=3,fy=3)
#cv2.imshow("kon",img)
#cv2.imshow("koncopy",copy)
#cv2.waitKey(0)

pt = []
rect = []
for con in biggest_contours:
    bbox = cv2.minAreaRect(con[1])  # (center(x, y), (width, height), angle of rotation) = cv.minAreaRect(points

    pts = np.int0(cv2.boxPoints(bbox))  # pt =[[x,y],[x,y],[x,y],[x,y],]
    rect.append(pts)
    for p in pts:
        pt.append(p)

    bbox_img = cv2.rectangle(img, pts[0], pts[2], (0, 0, 255), 2)

cv2.imshow("Contours",bbox_img)


#todo: Die ROIs auch sortieren!!! von links nach rechts!

# min and max values of x an y coordinates     img_cropped = [y0,y1,x0,x1]


img_cropped = gray[min([i[1] for i in pt]):max([i[1] for i in pt]), min([i[0] for i in pt]):max([i[0] for i in pt])]
cv2.imshow("big",img_cropped)

h_orig,w = gray.shape
cv2.imshow("gry",gray)
print(h_orig,w)
messung_und_xWert = []
for no in range(10):
    single_rec =  gray[min([i[1] for i in rect[no]]):max([i[1] for i in rect[no]]), min([i[0] for i in rect[no]]):max([i[0] for i in rect[no]])]
    #Bestimmung des Abstands vom unteren Rand (const. Linie) um diesen als Nullpunkt zu nehmen
    #print(rect[no])

    dist_untere_linie = h_orig - max([i[1] for i in rect[no]])
    #print(dist_untere_linie)

    rot = cv2.rotate(single_rec,cv2.ROTATE_90_CLOCKWISE)
    h,w = rot.shape
    #print(h,w)
    haelfte = int(round(h/2))
    cut20 = int(h/10)
    # cuttign off the top and bottom rows AND the left side with a const. Value: 30
    # oben und unten werden die kanten abgeschnitten um die Threshold berechnung in OTSU nicht negativ zu beeinflussen
    # falls der untere Rand des Reagenzglases nicht gerade aufgeroppt wurde und ein dunkle/heller rand bestehen bleibt
    # entsteht hier eine Kante und entsprechend stoppt die Messung bei einem Wert von 1
    rot = rot[0+cut20:h-cut20,30:w]     #<- 30 untere Pixel abschneiden
    #print(rot.shape)
    cv2.imshow("rot",rot)



    #cv2.imshow("big_rectangle", img_cropped)
    #cv2.imshow("single_rectangle", rot)

    blur = cv2.GaussianBlur(rot, (5, 5), 0)
    ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #thresh = cv2.adaptiveThreshold(rot, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9,1)
    #ret2,thresh= cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret2,thresh= cv2.threshold(rot,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(rot, 181, 255, cv2.THRESH_BINARY)
    cv2.imshow("AddaptiveThresh",thresh)

    # print(single_line)
    # for i in single_line:
    #     print(int(i)-(int(i+1)))
    # print("######")
    single_line = thresh[haelfte:haelfte + 1, 0:w]
    edges = cv2.Canny(single_line,30,100)
    edges_big = cv2.Canny(rot,20,80)   # edgedetection setzt auch immer ein thresholding voraus zum binarisieren!
    #edges_big= cv2.dilate(edges_big,(3,3),iterations=2)
    #print(edges)

    messwert_von_unten = np.where(edges==255)[1].min()+30+dist_untere_linie
    x_wert_des_ROI = min([i[0] for i in rect[no]])

    messung_und_xWert.append([x_wert_des_ROI,messwert_von_unten])
    cv2.imshow("Cann",edges)
    cv2.imshow("CannBig",edges_big)

print(messung_und_xWert)
messung_und_xWert= sorted(messung_und_xWert, key=lambda x:x[0])
print(messung_und_xWert)
x=1
for i in messung_und_xWert:
    print(f"###Messung:{x}###{i[1]}") #erste Wert von rechts
    x +=1
#todo: Problem: durch die neue Einstellung und Messung der untersten Kante kann es dazu kommen, dass sich die Nulllinie verschiebt
# Lösung -> Die unterste Kante des Bildes immmmmmmer hinzu ziehen!!! somit hat man, insofern nichts wackelt immer eine const. Nulllinie!!!


#cv2.imshow("single_line", single_line)



h,w =rot.shape
xpoints = np.array([i for i in range(0,w)])
ypoints = np.array([i for i in rot[haelfte]])
plt.plot(xpoints, ypoints)
#plt.scatter(xpoints, ypoints, s=3)
plt.show()
plt.close()
cv2.imshow("Contours",bbox_img)

cv2.waitKey(0)

