from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
from text_to_speech import text_to_speech
import cv2

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < 0.5:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	return (rects, confidences)

def text_recognitionn():
	
	image = cv2.imread('ocr_images/image1.jpg')
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = origW / float(newW)
	rH = origH / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet('frozen_east_text_detection.pb')

	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	results = []

	for (startX, startY, endX, endY) in boxes:
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		dX = int((endX - startX))
		dY = int((endY - startY))

		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		roi = orig[startY:endY, startX:endX]

		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		results.append(((startX, startY, endX, endY), text))

	results = sorted(results, key=lambda r:r[0][1])

# loop over the results
	for ((startX, startY, endX, endY), text) in results:
		print("OCR TEXT")
		print("========")
		print("{}\n".format(text))

		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		print(text)
		text_to_speech(text)
		text_to_speech("text ended")

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
