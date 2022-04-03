import cv2
import numpy as np
from matplotlib import pyplot as plt

def color_filter(image, Grenze_unten_hsv, Grenzen_oben_hsv, bild_anzeigen=False, delay=3000):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_range = np.array(
        Grenze_unten_hsv)  # hue(Farbwert):[0,179], saturation(Farbsättigung, grau-reine Farbe) [0,255] value(Hellwert,Dunkelstufe,keine Helligkeit - volle Helligkeit) [0,255].
    upper_range = np.array(Grenzen_oben_hsv)  # [130,255,255]
    mask = cv2.inRange(image_hsv, lower_range, upper_range)
    mask_on_orig = cv2.bitwise_and(image, image, mask=mask)
    if bild_anzeigen == True:
        cv2.imshow('Filtered', mask_on_orig)
        cv2.waitKey(delay)
    return mask, mask_on_orig


def histo_HSV(img,channel="h",size_factor=1):
    """This Function works only for BGR or HSV images
    Hue            -     Farbwert
    Saturation     -     Sättigung
    Value          -     Helligkeit
    """

    cv2.imshow("Original",cv2.resize(img,(0,0),fx=size_factor, fy=size_factor,interpolation=cv2.INTER_AREA))

    try:
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print("converting to HSV img")
    except:
        print("it is a HSV img")
        pass  # if it is a HSV-img

    hue,saturation,value = cv2.split(img)

    cv2.imshow("Hue",cv2.resize(hue,(0,0),fx=size_factor, fy=size_factor,interpolation=cv2.INTER_AREA))
    cv2.imshow("Saturation",cv2.resize(saturation,(0,0),fx=size_factor, fy=size_factor,interpolation=cv2.INTER_AREA))
    cv2.imshow("Value",cv2.resize(value,(0,0),fx=size_factor, fy=size_factor,interpolation=cv2.INTER_AREA))

    if channel == "h":
        plt.title('Hue')
        plt.xlabel('Bins')
        plt.ylabel('sum of pixels')
        plt.hist(hue.ravel(), 256, [0, 256])
        plt.show()
    elif channel == "s":
        plt.title('Saturation')
        plt.xlabel('Bins')
        plt.ylabel('sum of pixels')
        plt.hist(saturation.ravel(), 256, [0, 256])
        plt.show()
    elif channel == "v":
        plt.title('Value')
        plt.xlabel('Bins')
        plt.ylabel('sum of pixels')
        plt.hist(value.ravel(), 256, [0, 256])
        plt.show()

    cv2.waitKey(0)

def biggest_box(binaer, img,numBigCon = 10):
    contours, hierarchy = cv2.findContours(image=binaer, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,lineType=cv2.LINE_AA)
    contour_sizes = [(cv2.contourArea(contour),contour) for contour in contours]
    #sort = [for con in contour_sizes[0]]
    contour_sizes.sort(key=lambda x:x[0])

    biggest_contours = contour_sizes[-numBigCon:]


    return biggest_contours

