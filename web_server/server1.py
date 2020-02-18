from flask import Flask
from flask import request
import os
app = Flask(__name__)

@app.route('/currency_detection',methods = ["POST"])
def curreny_detection():
    #print(request.body)
    picture=request.files['files']
    # print(picture)
    # save_file=os.path.join('testing_image','image1.jpg')
    # picture.save(save_file)
    picture.save(picture.filename)
    return "hello"


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8090)