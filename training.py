import os

import cv2
import numpy as np


eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50) # num_components=50, threshold=5
fisherface = cv2.face.FisherFaceRecognizer_create() # num_components=50, threshold=5
lbph = cv2.face.LBPHFaceRecognizer_create()

def get_image_with_id():
	paths = [os.path.join('photos', f) for f in os.listdir('photos')]
	faces = []
	ids = []

	for path in paths:
		image_face = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
		_id = int(os.path.split(path)[-1].split('.')[1])
		ids.append(_id)
		faces.append(image_face)
		#cv2.imshow('Face', image_face)
		#cv2.waitKey(1000)
	return np.array(ids), np.array(faces)


ids, faces = get_image_with_id()

print('Treinando...')

eigenface.train(faces, ids)
eigenface.write('./materials/EigenFaceClassifier.yml')

fisherface.train(faces, ids)
fisherface.write('./materials/FisherFaceClassifier.yml')

lbph.train(faces, ids)
lbph.write('./materials/LBPHFaceClassifier.yml')