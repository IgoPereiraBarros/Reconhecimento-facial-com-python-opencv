import os

import cv2
import numpy as np
from PIL import Image



eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def get_image_with_id():
	paths = [os.path.join('train', f) for f in os.listdir('train')]
	faces = []
	ids = []

	for path in paths:
		imageface = Image.open(path).convert('L') # converte para escala de cinza
		imageNP = np.array(imageface, 'uint8')
		_id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
		ids.append(_id)
		faces.append(imageNP)

	return np.array(ids), faces

ids, faces = get_image_with_id()

print('Treinando...')

eigenface.train(faces, ids)
eigenface.write('EigenFaceClassifier.yml')

fisherface.train(faces, ids)
fisherface.write('FisherFaceClassifier.yml')

lbph.train(faces, ids)
lbph.write('LBPHFaceClassifier.yml')