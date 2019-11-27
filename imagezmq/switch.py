from gpiozero import Button
from signal import pause
from client import client
from time import sleep
from text_to_speech import text_to_speech
from picamera import PiCamera
from currency_detection import currency_detection
from client import client
from text_recognition import text_recognitionn
from time import sleep

client_button=Button(2) #grey
ocr_button=Button(3) #white
distance_button=Button(20)  #black
currency_button=Button(21) #purple


text_to_speech("device started")
      
def client_action():
    try:
        print('object detection started...')
        text_to_speech('Object detection Started')
        client()
        print('object detection closed')
    finally:
        text_to_speech('object detection closed')
        
    
def ocr_action():
    text_to_speech('OCR Started')
    camera = PiCamera()
    try:
        camera.capture('/home/pi/Desktop/vision_guide/imagezmq/ocr_images/image1.jpg')
        camera.close()
        print('image captured')
    finally:
        camera.close()
        text_recognitionn()
        text_to_speech('OCR closed')
    
#def distance_action():
 #   print('distance')
    
#def distance_exit():
 #   print('distance exit')
    
def currency_action():
    text_to_speech('Currency Detection Started')
    camera = PiCamera()
    try:
        camera.capture('/home/pi/Desktop/vision_guide/imagezmq/currency_images/image1.jpg')
        camera.close()
        print('image captured')
    finally:
        camera.close()
        currency_detection()
        text_to_speech('Currency Detection closed')
        #exit(0)
    
while True:
    client_button.when_pressed=client_action
#client_button.when_released=client_exit

    ocr_button.when_pressed=ocr_action
#ocr_button.when_released=ocr_exit

    currency_button.when_pressed=currency_action
#currency_button.when_released=currency_exit

#distance_button.when_pressed=distance_action
#distance_button.when_released=distance_exit

    pause()
