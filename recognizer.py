import cv2



face_detector = cv2.CascadeClassifier('materials/haarcascade-frontalface-default.xml')

#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer.read('materials/EigenFaceClassifier.yml')

#recognizer = cv2.face.FisherFaceRecognizer_create()
#recognizer.read('materials/FisherFaceClassifier.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('materials/LBPHFaceClassifier.yml')

width, height = 220, 220

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

name = ''

while True:
	conected, image = camera.read()

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minSize=(150, 150))

	for (x, y, w, h) in detected_faces:
		image_face = cv2.resize(gray[y:y + h, x:x + w], (width, height))
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
		_id, confidence = recognizer.predict(image_face)

		if _id == 1:
			name = 'people 1'
		elif _id == 2:
			name = 'people 2'

		cv2.putText(image, str(name), (x, y + (h + 30)), font, 2, (0, 0, 255))
		cv2.putText(image, str(confidence), (x, y + (h + 70)), font, 2, (0, 0, 255))

	cv2.imshow('Recognizer', image)
	if cv2.waitKey(1) == ord('c'):
		break

camera.release()
cv2.destroyAllWindows()
