# Assumptions - 
1.) Incoming data to the api is assumed to be converted .npy files as mentioned in /test_api/test.py. 

2.) The image size is also assumed to be (256,256) which can be handled with minor code changes.

# MODEL TRAINING
- The model used is Unet. The model is trained from scratch with 300 images from DomainA with 20% as validation.

## Model -
- The model used to train is Unet. The model is trained from scratch with 300 images from DomainA with 20% of the dataset as validation.

## Model Results -
Results are domain A - 

![Alt text](https://github.com/coverahealthrecruiting/AI_ENG_Arpit_Maclay/blob/dev/results_images/result_1.png)

![Alt text](https://github.com/coverahealthrecruiting/AI_ENG_Arpit_Maclay/blob/dev/results_images/result2.png)

![Alt text](https://github.com/coverahealthrecruiting/AI_ENG_Arpit_Maclay/blob/dev/results_images/result3.png)

Color augmentation and contrast/brightness were adjusted in Domain A in order to reduce the domain difference between domain A and domain B but the model did not perform for domain B at all.

## Model Evalutaion
Dice score can be seen in the figure to improve upto 90 % while the training loss and validation loss plummets.

![Alt text](https://github.com/coverahealthrecruiting/AI_ENG_Arpit_Maclay/blob/dev/results_images/train_500_dataset.png)



# MODEL DEPLOYMENT
## Model/API Monitoring
There are 2 apis in the main file 
- /get_prediction - predicts the masks and returns [user_id','count_hits','api_name','batch_size','execution_time'] to the client.
- /get_all - return all the data in the sqlite database which can be processed to get desired monitoring variables.

An external library is used to create dashboard of all the api calls and model performance.

## To run Training
1.) Copy dataset into data.py
2.) Uncomment function calls in train.py
``` 
python train.py
```
## In order to run API server
Running main file will database.db and mydb.db for dashboard and custom apis SQL tables.
```
python
from main import db
db.create_all()
python main.py
```
These commands will start a server ready to accept requests.
The dashboard can be seen at http://localhost/dashboard with username - admin and password - admin.

## In order to test custom APIs
Custom api test file is saved in /test_api in order to run test apis after runnin the server run following commands to make client calls.
```
cd test_api
python test.py
```
Sample output from the api call -
```
 (user, 1, '/get_prediction', 72, 15.873160362243652)]
 ```

