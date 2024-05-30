import cv2
# from google.colab import output
#from google.colab.patches import cv2_imshow
import numpy as np
from time import sleep

width_min=80
height_min=80 

offset=10
pos_line=700

delay= 600
detec = []
car= 0

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('asset/IMG_3262.mp4')
subtraction = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    time = float(1/delay)
    sleep(time) 
    
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    
    img_sub = subtraction.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expand = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    expand = cv2.morphologyEx (expand, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(expand,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (30, pos_line), (1200, pos_line), (255,127,0), 3)
    # cv2.line(frame1, (200,0), (200,1000), (255,0,0), 2 )
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_outline = (w >= width_min) and (h >= height_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)   
        
        center = pega_centro(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_line+offset) and y>(pos_line-offset):
                car+=1
                cv2.line(frame1, (25, pos_line), (1200, pos_line), (0,127,255), 3)  
                detec.remove((x,y))
                print("helm is detected : "+str(car))        
       
    cv2.putText(frame1, "Kendaraan Lewat : "+str(car), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Detector",expand)
    cv2.imshow("Video Original" , frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()