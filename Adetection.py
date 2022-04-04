import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
#from basicFunctions import*

#########Params########

###########################
img = cv2.imread("Assets/3/140322_9_edited.png")

def measuring(img,
            scaling = 1,
            showing_orig = False,
            showing_orig_binaer = False,
            showing_orig_contours = False,
            showing_bbox_reagenzglaeser = False,
            showing_single_Reagenzglas = True,
            showing_single_Reagenzglas_binaer = False,
            showing_edges_of_Reagenzglas = True,
            plot_grayscales_of_edge = False,
            waitKey_for_showing_img = 0):

    img = cv2.resize(img,(0,0),fx=scaling,fy=scaling)

    if showing_orig == True: cv2.imshow("orig",img)

    ########## Wandlung in Graustufen, Glättung, und Binarisierung
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, binaer = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if showing_orig_binaer == True: cv2.imshow("Original_Binaer_OTSU",binaer)


    ########## Detection der 10 größten Kontouren des Binaerbilds
    def biggest_box(binaer, img,numBigCon = 10):
        contours, hierarchy = cv2.findContours(image=binaer, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour),contour) for contour in contours]
        contour_sizes.sort(key=lambda x:x[0])
        biggest_contours = contour_sizes[-numBigCon:]

        return biggest_contours

    img_copy_con = img.copy()

    biggest_contours = biggest_box(binaer,img=img_copy_con,numBigCon=10)

    if showing_orig_contours == True:
        for i in biggest_contours:
            cv2.drawContours(image=img_copy_con, contours=i[1], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Original_Kontouren",img_copy_con)

    ########## Ausschneiden von rechteckigen ROI in denen die Reagenzgläser enthalten sind

    rect = []
    for con in biggest_contours:
        bbox = cv2.minAreaRect(con[1])  # (center(x, y), (width, height), angle of rotation) = cv.minAreaRect(points
        pts = np.int0(cv2.boxPoints(bbox))  # pt =[[x,y],[x,y],[x,y],[x,y],]
        rect.append(pts)

        bbox_img = cv2.rectangle(img, pts[0], pts[2], (0, 0, 255), 2)

    if showing_bbox_reagenzglaeser == True:  cv2.imshow("ROIs_der_Reagenzglaeser",bbox_img)



    # min and max values of x an y coordinates     img_cropped = [y0,y1,x0,x1]

    h_orig,w = gray.shape

    messung_und_xWert = []
    for no in range(10):
        # ausschneiden des ROI um das zu messende Reagenzglas mit den Koordinaten aus der Konturerkennung und den um die
        # Konturen gezeichneten Rechtecken
        single_rec =  gray[min([i[1] for i in rect[no]]):max([i[1] for i in rect[no]]), min([i[0] for i in rect[no]]):max([i[0] for i in rect[no]])]
        #   ->  weitere Möglichkeite die passenden Koordinaten zu finden: min(rect[no], key=lambda x:x[1])][0][1]
        # Bestimmung des Abstands vom unteren Rand (const. Linie) um diesen als Nullpunkt zu nehmen
        dist_untere_linie = h_orig - max([i[1] for i in rect[no]])

        # Rotation um den Abstand einfacher zu messen (Zeilen lassen sich einfacher auslesen als Spalten)
        rot_single = cv2.rotate(single_rec,cv2.ROTATE_90_CLOCKWISE)
        h,w = rot_single.shape
        haelfte = int(round(h/2))       # Bestimmung der mittleren Zeile um an ihr die Messung vorzunehmen
        cut20 = int(h/10)               # Abschneiden der unteren und oberen Zeilen um bei der Schwellenwertbildung(Threshhold)
                                        # Einfluss von dunklen Kanten zu haben nicht zum Reagenzglas selbst gehören
        # cuttign off the top and bottom rows AND the left side with a const. Value: 30
        # oben und unten werden die kanten abgeschnitten um die Threshold berechnung in OTSU nicht negativ zu beeinflussen
        # falls der untere Rand des Reagenzglases nicht gerade aufgeroppt wurde und ein dunkle/heller rand bestehen bleibt
        # entsteht hier eine Kante und entsprechend stoppt die Messung bei einem Wert von 1
        rot_single = rot_single[0+cut20:h-cut20,30:w]     #<- 30 untere Pixel abschneiden
        if showing_single_Reagenzglas == True: cv2.imshow("Rotiertes Reagenzglas",rot_single)

        ######## Die eigentliche Messung

        # 1. Glättung und Binarisierung
        blur = cv2.GaussianBlur(rot_single, (5, 5), 0)
        ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if showing_single_Reagenzglas_binaer == True: cv2.imshow("Thresh_OTSU",thresh)

        # 2. Auschneiden einer einzigen Zeile zur Messung, Kantendetektion an Übergang(0 zu 255)

        #todo: evtl. mehrere Zeilen nehmen und den Mittelwert/Median bilden?


        single_line = thresh[haelfte:haelfte + 1, 0:w]
        edges = cv2.Canny(single_line,30,100)
        edges_big = cv2.Canny(thresh,20,80)   # edgedetection setzt auch immer ein thresholding voraus zum binarisieren!
        if showing_edges_of_Reagenzglas == True:
            cv2.imshow("Edges_of_single_Reagenzgls",edges_big)

        # 3. die eigentlichte Messung + addition der abgeschnittenen 30 Pixel und dem Abstand zu untersten Kante des Bildes
        messwert_von_unten = np.where(edges==255)[1].min()+30+dist_untere_linie
        x_wert_des_ROI = min([i[0] for i in rect[no]])

        messung_und_xWert.append([x_wert_des_ROI,messwert_von_unten])



    messung_und_xWert= sorted(messung_und_xWert, key=lambda x:x[0])
    messwerte = np.delete(messung_und_xWert, [0], 1)            # entfernen der X-kooridanten, die Nummer der Reagenzglases entspricht dem Index



    if plot_grayscales_of_edge == True:
        h,w =rot_single.shape
        xpoints = np.array([i for i in range(0,w)])
        ypoints = np.array([i for i in rot_single[haelfte]])
        plt.plot(xpoints, ypoints)
        #plt.scatter(xpoints, ypoints, s=3)
        plt.show()
        plt.close()

    cv2.waitKey(waitKey_for_showing_img)

    return [messwerte,img,binaer,img_copy_con,bbox_img,rot_single,thresh,edges_big]



mess=measuring(img)
print(mess[0])