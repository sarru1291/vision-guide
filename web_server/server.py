from flask import Flask
from utils import *
from matplotlib import pyplot as plt

import subprocess
from gtts import gTTS
import os

app = Flask(__name__)

@app.route('/currency_detection',methods = ["GET"])
def curreny_detection():
    #print(request.data)
    print('hellow world')
   
    max_val = 8
    max_pt = -1
    max_kp = 0

    orb = cv2.ORB_create()

    #test_img = read_img('files/test_100_2.jpg')
    #test_img = read_img('files/test_50_2.jpg')
    #test_img = read_img('testing_image/2000_1.jpg')
    #test_img = read_img('files/test_100_3.jpg')
    #test_img = read_img('files/test_20_4.jpg')

    # original = resize_img(test_img, 0.4)
    # display('original', original)
    test_img=read_img('testing_image/image1.jpg')

    (kp1, des1) = orb.detectAndCompute(test_img, None)

    #training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']
    my_list= os.listdir('files/')
    training_set=[]
    for i in range(len(my_list)):
        training_set.append('files/'+my_list[i])

    print(training_set)
    for i in range(0, len(training_set)):
        train_img = cv2.imread(training_set[i])

        (kp2, des2) = orb.detectAndCompute(train_img, None)

        bf = cv2.BFMatcher()
        all_matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for (m, n) in all_matches:
            if m.distance < 0.789 * n.distance:
                good.append([m])

        if len(good) > max_val:
            max_val = len(good)
            max_pt = i
            max_kp = kp2

        print(i, ' ', training_set[i], ' ', len(good))

    if max_val != 8:
        print(training_set[max_pt])
        print('\n Max OCR Value ', max_val)

        train_img = cv2.imread(training_set[max_pt])
        img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
        
        note = str(training_set[max_pt])[6:-4] 
        res=""
        print('\nDetected denomination: Rs. ', end=" ")
        for i in range(len(note)):
            if note[i]=='_':
                break
            print(note[i],end='')
            res+=note[i]
        return str(res)
        
        print('\n')
        
        #(plt.imshow(img3), plt.show())
    else:
        print('No Matches Found!!!')
        return "No matches found"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8090)