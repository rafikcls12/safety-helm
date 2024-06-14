if (y1+h<(350+offset) and y1+h>(350-offset)) and text=="Tidak Pakai Helm":
                    tp_helm.append(terdata)
                    
            if (y1+h<(350+offset) and y1+h>(350-offset)) and text=="Pakai Helm":
                
                for data in range(len(isi_l)):
                    try:
                        x, y, w, h = isi_p[data]
                        motor = img[y:y+h, x:x+w]
                        lower_red = np.array([0, 200, 0], dtype = "uint8") 
                        upper_red= np.array([0, 255, 0], dtype = "uint8")
                        mask = cv2.inRange(motor, lower_red, upper_red)
                        number_pixel= np.count_nonzero(mask)
                        if number_pixel > 0:
                            counterph += 1
                            
                    except:
                        pass
                        cv2.line(img, (600,0),(600,600),(0,255,0),5 )
                        print(counter)