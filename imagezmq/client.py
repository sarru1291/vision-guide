import socket
import time
from gpiozero import Button
from text_to_speech import text_to_speech
import json
from imutils.video import VideoStream
import imagezmq
import subprocess
from text_to_speech import text_to_speech
#from distance_sensor import distance_sensor

#navigation_off=Button(3) #white
def client():
    sender = imagezmq.ImageSender(connect_to='tcp://192.168.43.64:5555')
    rpi_name=socket.gethostname()
    picam=VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    while True:
        image=picam.read()
        text=sender.send_image(rpi_name,image)
        stext=str(text,'utf-8')
        jtext=json.loads(stext)
        for item in jtext:
            print(item['name'])
            text_to_speech(item['name'])
            
    if cv2.waitKey(0) & 0xFF == ord('q'):
#    if navigation_off.when_pressed:
        cv2.destroyAllWindows()
        text_to_speech("object detection closed")
        exit(0)
        
        
        

    
