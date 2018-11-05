SUMMARY OF FILES
----------------

1 ### vision.py

Takes original images from camera and pass it to Google Vision API.
Google Vision API will return whatever it can label.
Save the JSON output from Google Vision in vision/ folder.

2 ### render.py

Take JSON output from vision/ folder and convert into XML format for labelImg program.
These are saved into folder labels/.
Also, render the JSON output onto images for manual inspection in render/ folder.

3 ### datas.py

Prepare training data using XML files from labels/ folder and original images from images/ folder.
Training data is pickled into datas.pickle.

4 ### datas-tests.py

Prepare training data similar to datas.py.
Get inputs from datas_tests/ folder.

5 ### train.py

Script for training YOLO model.

6 ### test.py

Script for testing trained model.

7 ### requirement.in

Contains all the third party libraries. Install using pip-tools.
