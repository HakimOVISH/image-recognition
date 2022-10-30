import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np


#feature extractor class :
class FeatureExtractor:

    def __init__(self):
        base_model = VGG16(weights="imagenet")
        self.model =  Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        #pass
        
    def extract(self, img):
        img = img.resize((224,224)).convert("RGB")
        a = image.img_to_array(img)      #to a numpy array
        a = np.expand_dims(a, axis=0)    # (p,w,h) -> (1,p,w,f)
        a = preprocess_input(a)          #pixel value
        feature = self.model.predict(a)[0]   #passing the image to the model
        return feature / np.linalg.norm(feature)  #normalizing
    