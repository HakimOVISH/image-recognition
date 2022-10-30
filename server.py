#from crypt import methods
#from cgitb import reset
from turtle import distance
from isort import file
import numpy as np
from PIL import Image     # or also pillow for pil : a python library for image processing (data contained in an image)
from feature_extraction import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path


app= Flask(__name__)

#here we are reading image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem+".jpg"))
features = np.array(features)



@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        file = request.files["query_img"]
        #here i save the image given by the user
        img = Image.open(file.stream)   # using Image from pillow
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":",".")+"_"+file.filename
        img.save(uploaded_img_path)  # i'll be saving the image query in the uploaded directory

        #here we are going to begin the search
        query = fe.extract(img)      #extract features from the img given
        distances = np.linalg.norm(features-query, axis=1) #distances to features

        ids = np.argsort(distances)[:40] # first or top 40 searches

        scores = [(distances[id], img_paths[id]) for id in ids]

        #print(scores) #to print difference between images in terms of distance

        return render_template("index.html", query_path=uploaded_img_path ,scores=scores)  #retrieve the image given and display it from uploaded and display the results
       #return "Success!!! image well received" 

    else:
        return render_template("index.html")


if __name__ =="__main__":
    app.run()