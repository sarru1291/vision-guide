from gpiozero import Button
from signal import pause
from client import client
from time import sleep
import os
from imutils.video import VideoStream
from text_to_speech import text_to_speech
from picamera import PiCamera
from currency_detection import currency_detection
#from currency_detection1 import currency_detection1
from client import client
from text_recognition import text_recognitionn
from time import sleep

navigation_button=Button(2) #grey
ocr_button=Button(20) #black
#distance_button=Button(20)  #black
#navigation_off=Button(3) #white
currency_button=Button(21) #purple

print('Welcome')
text_to_speech("Welcome RefreshedBits. You've started Vision guide.")
      
def navigation_action():
    try:
        print('object detection starts..')
        text_to_speech('Object detection Starts')
        client()
        print('object detection closed')
    finally:
        text_to_speech('object detection closed')

#def navigation_action_off():
    #picam=VideoStream(usePiCamera=False).stop()
    #text_to_speech('object detection closed')
    #print('navigation closed')
    
def ocr_action():
    text_to_speech('Optical Character Recognition Starts')
    camera = PiCamera()
    try:
        camera.capture('/home/pi/Desktop/vision_guide/imagezmq/ocr_images/image1.jpg')
        camera.close()
        print('image captured')
    finally:
        camera.close()
        text_recognitionn()
        text_to_speech('Optical Character Recognition Closed')
    
#def distance_action():
 #   print('distance')
    
#def distance_exit():
 #   print('distance exit')
    
def currency_action():
    text_to_speech('Currency Denomination Starts')
    camera = PiCamera()
    try:
        camera.capture('/home/pi/Desktop/vision_guide/imagezmq/currency_captured_images/image1.jpg')
        camera.close()
        print('image captured')
    finally:
        camera.close()
        currency_detection()
        #currency_detection1()
        text_to_speech('Currency Denomination closed')
        os.remove("currency_captured_images/image1.jpg")
        #exit(0)
    
while True:
    navigation_button.when_pressed=navigation_action
#navigation_button.when_released=client_exit
#    navigation_off.when_pressed=navigation_action_off

    ocr_button.when_pressed=ocr_action
#ocr_button.when_released=ocr_exit

    currency_button.when_pressed=currency_action
#currency_button.when_released=currency_exit

    #distance_button.when_pressed=distance_action
#distance_button.when_released=distance_exit

    pause()
