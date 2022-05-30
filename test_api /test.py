import requests
import numpy as np
import io
import zlib
import matplotlib.pyplot as plt


file_name = "dat.npy"
### processes the incoming volume and outputs its shape and completiton flags
resp_predictions = requests.post("http://127.0.0.1:5000/get_prediction", json = {'dat.npy':np.memmap(file_name, dtype='uint8', mode='r').tolist(),'user':'arpit'}) 
print(resp_predictions.content)

## gets all the queries from the database
resp = requests.get("http://127.0.0.1:5000/get_a") 
print(resp.content)

#get each hit on the api in order to monitor
counts = requests.get("http://127.0.0.1:5000/get_count_hits") 
print(counts.content)