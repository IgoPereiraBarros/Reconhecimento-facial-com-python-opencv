import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


classifier = cv2.CascadeClassifier('materials/haarcascade-frontalface-default.xml')
classifier_eye = cv2.CascadeClassifier('materials/haarcascade-eye.xml')

camera = cv2.VideoCapture(0)

samples = 1
number_samples = 25

_id = int(input('Informe seu indicador (valor inteiro): '))
width, height = 220, 220

while True:
	connected, image = camera.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logger.warning(np.average(gray))
	detected = classifier.detectMultiScale(gray, scaleFactor=1.1, minSize=(150, 150))

	for (x, y, w, h) in detected:
		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

		region = image[y:y + h, x:x + w]
		region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
		eye_detected = classifier_eye.detectMultiScale(region_gray)

		for (eye_x, eye_y, eye_w, eye_h) in eye_detected:
			cv2.rectangle(region, (eye_w, eye_h), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)

			if cv2.waitKey(1) & 0xFF == ord('s'):
				if np.average(gray) > 110:
					face = cv2.resize(gray[y:y + h, x:x + w], (width, height))
					cv2.imwrite('photos/people.' + str(_id) + '.' + str(samples) + '.jpg', face)
					logger.warning('[foto ' + str(_id) + '.' + str(samples) + ' capturada com sucesso]')
					samples += 1

	cv2.imshow('Face', image)

	if samples >= number_samples + 1:
		break

	if cv2.waitKey(1) == ord('c'):
		break

logger.warning('Faces capturadas com sucesso')
camera.release()
cv2.destroyAllWindows()