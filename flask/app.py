import os
from flask import Flask, app, request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input



app=Flask(__name__)
model=load_model("shipclassification.h5")


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')
@app.route('/index.html')
def home():
    return render_template("index.html")


@app.route('/result',methods=["GET","POST"])
def res():
    if request.method == "POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path. join(basepath, 'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(224,224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        # reshape data for the model
        pred = model.predict(img)
        pred=pred.flatten()
        pred = list(pred)
        m = max(pred)
        val_dict ={0:'Cargo',1:'Carrier', 2:'Cruise', 3:'Military', 4:'Tankers'}
        #print(val_dict[pred.index(m)])
        result=val_dict[pred.index(m) ]
        image_file = 'uploads/' + f.filename
        #print(result)
        return render_template('prediction.html',prediction=result, image_file=image_file)
    else:
        return render_template('prediction.html')
if __name__ =="__main__":
    app.run(debug=True)
    