from flask import Flask, request, jsonify, Response
from pytesseract import image_to_string
from PIL import Image
import base64
import sys 
from  io import BytesIO
from io import StringIO
import cv2
import updatedscan
import numpy as np
import pycurl
import urllib
import urllib.parse
from crop import crop

app = Flask(__name__)
portnum = '5000'
hostname = 'localhost'
@app.route("/xiao")
def integrationtest():
    return 'hi xiao'

@app.route("/john", methods=['GET']) 
def otherget(): 
    fh = open('output.txt','r').readlines()[0]
    param = [{'imageencoded': fh} , {'imagepath': 'scan_res.jpg'}]
    body = makeCurl(param[0], 'http://localhost:5000/' + '?' + urllib.parse.urlencode(param[0]))
    body = makeCurl({}, 'http://localhost:5000/deskewed')
    body = makeCurl({}, 'http://localhost:5000/crop')
    body = makeCurl(param[1], 'http://localhost:5000/ocr' + '?' + urllib.parse.urlencode(param[1]))
    #param = {'text': body} 
    #body = makeCurl(param, 'http://localhost:5000/regex' + '?' + urllib.parse.urlencode(param))
    return body

@app.route("/test", methods=['POST']) 
def e2etest(): 
    fh = request.json['data']
    param = [{'imageencoded': fh} , {'imagepath': 'scan_res.jpg'}]
    body = makeCurl(param[0], 'http://'+ hostname + ':'+ portnum + '/' + '?' + urllib.parse.urlencode(param[0]))
    body = makeCurl({}, 'http://'+ hostname + ':' + portnum + '/deskewed')
    body = makeCurl({}, 'http://'+ hostname + ':'+ portnum +'/crop')
    body = makeCurl(param[1], 'http://'+ hostname + ':'+ portnum + '/ocr' + '?' + urllib.parse.urlencode(param[1]))
    print('\n')
    print(body)
    param = {'text': body} 
    body = makeCurl(param, 'http://'+ hostname + ':'+ portnum + '/regex' + '?' + urllib.parse.urlencode(param))
    print(body)
    return body

def makeCurl(param, url):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    return buffer.getvalue()

@app.route("/", methods=['GET'])
def saveimage():
    imagestring = request.args['imageencoded']
    imagestring = imagestring.encode('utf-8')
    imageobj = Image.open(BytesIO(base64.b64decode(imagestring)))
    #imageobj = Image.open(BytesIO(base64.b64decode(imagestring))).rotate(-90)
    cv2.imwrite('tejpic.jpg', np.array(imageobj))
    updatedscan.scan(np.array(imageobj))
    return 'hi john'

@app.route("/deskewed")
def deskewimage(): 
    img = cv2.imread('deskewed.jpg')
    img = cv2.dilate(img, np.ones((2, 2)))
    newimgname = 'no_noise.jpg'
    cv2.imwrite(newimgname, img)
    return newimgname

@app.route("/crop")
def cropimage():
    crop('no_noise.jpg', "scan_res.jpg")
    return "scan_res.jpg"


@app.route("/ocr", methods=['GET'])
def OCRImage():
    imagelocation = request.args['imagepath']
    return image_to_string(Image.open(imagelocation), config="config")

@app.route("/regex")
def regex():
    imagestring = request.args['text'].encode('utf-8').split('\n') 
    toReturn = []
    subtotal = 0
    total = 0
    tax = 0
    for i in imagestring: 
        if '.' in i: 
            brokenstring = i.split(' ')
            key = ''
            for num in range(len(brokenstring) - 1):
                key += brokenstring[num] + ' '
            value = brokenstring[-1]

            if '$' in value: 
                value = value[1:]
            
            try: 
                value = value.replace('\'','.')
                value = float(value)
            except: 
                continue

            if 'Visa' in key: 
                continue

            if 'Subtot' in key:
                subtotal = value
                continue
            if 'Paymen' in key: 
                total = value
                continue
            if 'Tax' in key: 
                tax = value
                continue 
            if 'hange' in key:
                continue

            toReturn.append({'itemName': key, 'price': value})

    return jsonify({'items': toReturn, 'total': total, 'subtotal': subtotal, 'tax': tax})






@app.route("/", methods=['POST'])
def post():
    imagestring = request.json['data']
    a = BytesIO(base64.b64decode(imagestring))
    imageobj = Image.open(BytesIO(base64.b64decode(imagestring)))
    updatedscan.scan(np.array(imageobj), False)
    final_string = image_to_string(imageobj).encode('utf-8')
    return jsonify(final_string)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = portnum,  threaded=True)


