import cv2
import numpy as np
import time
import datetime

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('asset/IMG_3262.mp4')
net = cv2.dnn.readNetFromONNX("best5.onnx")
classes = ["Pakai Helm", "Motor", "Tidak Pakai Helm","q","qw","asd","asdsd","l","po","0","asdw","k","g","s"]

count=0
counter=0

counterm=0
counterph=0
countertph=0

def gen_frames():
    start_time = time.time()
    frame_id = 0
    offset=5
    global count, counter, counterm, counterph  # generate frame by frame from camera
    while True:
        img = cap.read()[1]
        if img is None:
            break
        img = cv2.resize(img, (1080,600))
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        detections = net.forward()[0]
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        dt = str(datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y "))
        cv2.putText(img, dt, (680,40),font, 0.7,(0,255,0),2,cv2.LINE_8)

        # cx,cy , w,h, confidence, 80 class_scores
        # class_ids, confidences, boxes

        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.1:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.3:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.3)
        isi_l=[]
        isi_p=[]
        m = "Motor"
        ph = "Motor Menggunakan Helm"
        tph = "Motor Tidak Menggunakan Helm"
        tp_helm = []
        for i in indices:
            terdata = datetime.datetime.now()
            x1,y1,w,h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)
            confi = "{:.2f}".format(conf)
            cv2.putText(img, confi, (x1,y1),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4,(0,0,0),1)
            if label == "Tidak Pakai Helm":
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,255),1)
            if label == "Pakai Helm":
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),1)
            if label == "Motor":
                isi_l.append(label)
                isi_p.append(boxes[i])
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),1)

            if (y1+h<(350+offset) and y1+h>(350-offset)) and text=="Motor":
                    
                    for data in range(len(isi_l)):
                        try:
                            x, y, w, h = isi_p[data]
                            motor = img[y:y+h, x:x+w]
                            motorr = cv2.cvtColor(motor, cv2.COLOR_BGR2HSV)

                            lower_red = np.array([0, 100, 100], dtype = "uint8")
                            upper_red= np.array([10, 255, 255], dtype = "uint8")
                            lower_g = np.array([35, 50, 50], dtype = "uint8")
                            upper_g = np.array([85, 255, 255], dtype = "uint8")
                            mask = cv2.inRange(motorr, lower_red, upper_red)
                            maskk = cv2.inRange(motorr, lower_g, upper_g)
                            number_pixel= np.count_nonzero(mask)
                            number_pixell= np.count_nonzero(maskk)
                            print(number_pixel)
                            print(number_pixell)
                            if number_pixel > 0:
                                counter += 1
                                counterm += 1
                            if number_pixell > 0:
                                counterph += 1
                                counterm += 1 
                        except:
                            pass 
                            print(counter)
            # if (y1+h<(350+offset) and y1+h>(350-offset)) and text=="Tidak Pakai Helm":
            #         tp_helm.append(terdata)
                    
            # if (y1+h<(350+offset) and y1+h>(350-offset)) and text=="Pakai Helm":
                
            #     for data in range(len(isi_l)):
            #         try:
            #             x, y, w, h = isi_p[data]
            #             motor = img[y:y+h, x:x+w]
            #             lower_red = np.array([0, 200, 0], dtype = "uint8") 
            #             upper_red= np.array([0, 255, 0], dtype = "uint8")
            #             mask = cv2.inRange(motor, lower_red, upper_red)
            #             number_pixel= np.count_nonzero(mask)
            #             if number_pixel > 0:
            #                 counterph += 1
                            
            #         except:
            #             pass
            #             cv2.line(img, (600,0),(600,600),(0,255,0),5 )
            #             print(counter)
            cv2.putText(img, str(counterm), (500,80),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0),2)
            cv2.putText(img, str(counterph), (500,105),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0),2)
            cv2.putText(img, str(counter), (500,130),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0),2)

            cv2.putText(img, m, (10,80),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0),2)
            cv2.putText(img, ph, (10,105),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0),2)
            cv2.putText(img, tph, (10,130),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0),2)

            #cv2.line(img, (0,400),(1000,400),(200,200,0),4)
            cv2.line(img, (0,350+offset),(1000,350+offset),(0,255,0),4)
            cv2.line(img, (0,350-offset),(1000,350-offset),(0,255,0),4)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            # yield (b'--frame\r\n'
            #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            waktu = time.strftime("%H:%M:%S")
            tanggal=time.strftime("%m/%d/%Y")
        frame_id += 1
            
            # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = frame_id / elapsed_time
            
            # Print FPS on the image
        cv2.putText(img, f"FPS: {fps:.2f}", (700,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,cv2.LINE_8)
        cv2.imshow('Video', img)   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
if __name__ == '__main__':
    gen_frames()
    # cv2.imshow("VIDEO",img)
    # k = cv2.waitKey(1)
    # if k == ord('q'):
    #     break
