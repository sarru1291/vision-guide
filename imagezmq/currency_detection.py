from utils import *
from matplotlib import pyplot as plt
import cv2
import subprocess
import os
from text_to_speech import text_to_speech
from gtts import gTTS

def currency_detection():
	#print('starting')
	max_val = 8
	max_pt = -1
	max_kp = 0

	orb = cv2.ORB_create()
	test_img = read_img('currency_captured_images/image1.jpg')

	original = resize_img(test_img, 0.4)
	(kp1, des1) = orb.detectAndCompute(test_img, None)
	
	#print('before')
	#training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']
	my_list=os.listdir('files/')
	training_set=[]
	for i in range(len(my_list)):
		training_set.append('files/'+my_list[i])
	#print('train')
	for i in range(0, len(training_set)):
	# train image
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
		print('\nMatch value: ', max_val)

		train_img = cv2.imread(training_set[max_pt])
		img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
		note = str(training_set[max_pt])[6:-4]
		print('\nDetected denomination: Rs. ', end="")
		for i in range(len(note)):
			if note[i]=='_':
				break
			print(note[i],end='')
		print('\n')
		text_to_speech("It is %s rupees" % note)
		#text_to_speech("Currency detection closed")
		
	else:
		print('No matches found')
		text_to_speech("No Matches found")
	
