import os
import logging

import cv2
import numpy as np
from PIL import Image



logger = logging.getLogger(__name__)

face_detector = cv2.CascadeClassifier('../materials/haarcascade-frontalface-default.xml')

#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer.read('EigenFaceClassifier.yml')

#recognizer = cv2.face.FisherFaceRecognizer_create()
#recognizer.read('FisherFaceClassifier.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('LBPHFaceClassifier.yml')

total_hit = 0
percent_hit = 0.0
total_confidence = 0.0

paths = [os.path.join('test', f) for f in os.listdir('test')]

for path in paths:
	image_face = Image.open(path).convert('L')
	image_faceNP = np.array(image_face, 'uint8')
	detected_faces = face_detector.detectMultiScale(image_faceNP)

	for (x, y, w, h) in detected_faces:
		id_predicted, confidence = recognizer.predict(image_faceNP)
		current_id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
		logger.warning(str(current_id) + ' foi classificado como ' + str(id_predicted) + ' - ' + str(confidence))
		
		if id_predicted == current_id:
			total_hit += 1
			total_confidence += confidence

		cv2.rectangle(image_faceNP, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.imshow("Face", image_faceNP)
		cv2.waitKey(1000)


percent_hit = (total_hit / 30) * 100
total_confidence /= total_hit

logger.warning('Percentual de acertos ' + str(percent_hit))	
logger.warning('Total confiança ' + str(total_confidence))	
