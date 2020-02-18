import RPi.GPIO as GPIO
import time
from  text_to_speech import  text_to_speech

try:
                  GPIO.setmode(GPIO.BOARD)

                  PIN_TRIGGER = 7
                  PIN_ECHO = 11

                  GPIO.setup(PIN_TRIGGER, GPIO.OUT)
                  GPIO.setup(PIN_ECHO, GPIO.IN)
                  print("starting distance sensor...")
                  text_to_speech("starting distance sensor")
                  while True:

                        GPIO.output(PIN_TRIGGER, GPIO.LOW)


                        time.sleep(2)


                        GPIO.output(PIN_TRIGGER, GPIO.HIGH)

                        time.sleep(0.00001)

                        GPIO.output(PIN_TRIGGER, GPIO.LOW)

                        while GPIO.input(PIN_ECHO)==0:
                              pulse_start_time = time.time()
                        while GPIO.input(PIN_ECHO)==1:
                              pulse_end_time = time.time()

                        pulse_duration = pulse_end_time - pulse_start_time
                        distance = round(pulse_duration * 17150, 2)
                        print("distance: "+str(distance) +"cm")
                        text_to_speech(str(distance))
                        text_to_speech("cm")

finally:
                  GPIO.cleanup()
