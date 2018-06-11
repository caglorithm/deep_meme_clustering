import imagehash
import hashlib
import tqdm 
import numpy as np

from PIL import Image

from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import decode_predictions
from keras import backend as K
from keras.preprocessing import image as image


from keras.models import Model


def is_image(filename):
    f = filename.lower()
    return f.endswith(".png") or f.endswith(".jpg") or \
    f.endswith(".jpeg") or f.endswith(".bmp")

def return_phash(filename):
	hashfunc = imagehash.phash

	img = Image.open(filename)
	if img is not None:
		img_hash = hashfunc(img, hash_size=12) # phash
		#img_hash = hashfunc(img, hash_size=4) # whash
	return img_hash

def return_phashes(files):
	phashes = []
	for f in tqdm.tqdm(files, total=len(files)):
		phash = str(return_phash(f))
		phashes.append(phash)
	return phashes

def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()



# DEEP LEARNING STUFF

def get_model(modelname='ResNet50'):
	if modelname == 'ResNet50':
		model = get_model_ResNet50()
	else:
		raise Exception("modelname not found")
	return model

def get_model_ResNet50():
	model = ResNet50(weights='imagenet', include_top=True)
	getFingerprint = K.function([model.layers[0].input],
                                  [model.layers[-2].output])
	return model, getFingerprint

def fingerprints(files, model, getFingerprint, size=(224,224), modelname='ResNet50'):

    all_results = {}
    fps = []
    predictions = []
    labels = []
    for fn in tqdm.tqdm(files, total=len(files)):
        fp, prediction, label = fingerprint(fn, model, getFingerprint, size, modelname=modelname)
        fps.append(fp)
        predictions.append(predictions)
        labels.append(label)

        if len(fp) > 0:
            all_results[fn] = fp

    #return dict((fn, fingerprint(fn, model, size)) for fn in files)
    #return all_results
    return fps, predictions, labels

def fingerprint(fn, model, getFingerprint, size, modelname='ResNet50'):
	img = image.load_img(fn, target_size=size)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	# predict labels (last layer)
	prediction = model.predict(x)
	labels = decode_predictions(prediction, top=5)[0]
	labels = [[l[1], l[2]] for l in labels]
	# predict fingerprint (second to last layer)
	fingerprint = getFingerprint([x])[0].flatten()

	return fingerprint, prediction, labels