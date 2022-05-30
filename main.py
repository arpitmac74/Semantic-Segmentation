from calendar import c
import json
from flask import Flask, jsonify, request
from flask_restful import Api,Resource
from flask_sqlalchemy import SQLAlchemy
from predict import predict
from unet.unet_model import UNet
import torch
import numpy as np
import flask_monitoringdashboard as dashboard
from flask_marshmallow import Marshmallow 
from marshmallow import Schema
import time


from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'npy'}

def allowed_file(filename):
    """
    function to check if the received file is in allowed format
    """
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

##init db
db = SQLAlchemy(app)

#Init ma
ma = Marshmallow(app)

#Product class/Model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    user_name = db.Column(db.String(100),unique = True)
    count = db.Column(db.Integer)
    api_name = db.Column(db.String)
    batch_size = db.Column(db.Integer)
    time_exec = db.Column(db.Integer)

    def __init__(self, user_name,count,api_name,batch_size,time_exec):
        self.name = user_name
        self.count = count 
        self.api_name = api_name
        self.batch_size = batch_size
        self.time_exec = time_exec

#Schema
class Schemas(ma.Schema):
    class Meta:
        fields = ('id','count','api_name','batch_size','time_exec')

#init schema 
schema_p = Schemas()
schema_users =  Schemas(many = True)

# dashboard
dashboard.config.init_from(file='config.cfg')
dashboard.bind(app)

counter = 1

MODEL_PATH = "./model/unet_trained.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels= 1,n_classes = 2, bilinear=True)
if device != 'cpu':
    model.to(device=device)
model.load_state_dict(torch.load(MODEL_PATH,map_location = torch.device(device)))

###prediction api
@app.route('/get_prediction',methods = ['POST'])
def get_prediction():
    if request.method == 'POST':
        start_time = time.time()
        data = request.json
        count = 0
        api_name = request.path
        user_name = data[list(data.keys())[1]]
        file = list(data.keys())[0]
        if file is None or file == "":
            return jsonify({'error':'no file'})
        if not allowed_file(file):
            return jsonify({'error':'format not supported'})
        
        try:
            img_input = np.asarray(data['dat.npy'])
        except:
            return jsonify({'error':'data format error in received file'})
        
        try:
            img_input = img_input.reshape(-1, 256, 256, 1)
            batch_size = img_input.shape[0]
            result = predict(model,img_input,device= device)
            result = result.numpy() ###final output result #####
            time_exec = time.time() - start_time
        except:
            return jsonify({'error':'error during prediction'})
        new = Product(user_name,count,api_name,batch_size,time_exec)
        db.session.add(new)
        db.session.commit()
        return schema_p.jsonify(new)


# request to get all the entries
@app.route('/get_a',methods = ['GET'])
def get_all():
    all_queries = Product.query.all()
    result = schema_users.dump(all_queries)
    return jsonify(result)


@app.route("/get_count_hits")
def main():
    global counter
    counter += 1
    return str(counter)

if __name__ == "__main__":
    app.config['TRACK_USAGE_USE_FREEGEOIP'] = False
    app.config['TRACK_USAGE_INCLUDE_OR_EXCLUDE_VIEWS'] = 'include'
    app.run(debug = True)
