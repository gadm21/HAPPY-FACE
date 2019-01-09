# AI model training 

## NOTE: for Vehicle Recognition module (can be applied to other modules)

1. Image sources

Store label_img XML in following folders format: datas/{folder_name}/labels/{}.xml
Store images in following folders format: datas/{folder_name}/images/{}.jpg


2. Run datas.py to prepare training data.

This will process the images:

- resize
- change to gray scale
- etc

This will prepare the labels:

- convert label into appropriate format for TensorFlow.

The output will be a file called: datas.pickle.


3. Run train.py to train data in datas.pickle.

- Model files will be generated in models/ folder.


4. Model tuning.

- Check the training results.
- Depending on the validation accuracy / loss … tune the model parameters as required.


# SUMMARY OF FILES

## vision.py

THIS IS CURRENTLY NOT USED

Takes original images from camera and pass it to Google Vision API.
Google Vision API will return whatever it can label.
Save the JSON output from Google Vision in vision/ folder.

## render.py

Take JSON output from vision/ folder and convert into XML format for labelImg program.
These are saved into folder labels/.
Also, render the JSON output onto images for manual inspection in render/ folder.

## datas.py

Prepare training data using XML files from labels/ folder and original images from images/ folder.
Training data is pickled into datas.pickle.

## datas-tests.py

Prepare training data similar to datas.py.
Get inputs from datas_tests/ folder.

## train.py

Script for training YOLO model.

## test.py

Script for testing trained model.

7 ### requirement.in

Contains all the third party libraries. Install using pip-tools.
