import requests
import json

url="http://192.168.43.64:8090/currency_detection"
files={'media':open('currency_captured_images/image1.jpg','rb')}
r=requests.post(url,files=files)
print(r)
