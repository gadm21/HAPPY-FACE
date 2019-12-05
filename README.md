# How to setup MySQL
[link: https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-18-04]

# Create Database + Table (Only First Time)
Run the command to create database and related table
```
sudo mysql -u <username> -p < setup.sql  
```
# Access Database
```
sudo mysql -u <username> -p

# Select database
use face;

# Check all tables in selected database
show tables;

# Retrieve data from table
select * from Demographic;
```

# How to run the code
```
# Setup aws credentials  
aws configure  

# Installation   
pip3 install -r requirements.txt  

# Run (Please make sure the MYSQL database is created)  
cd multi-camera 
python3 main.py  
```

then the cropped image will save in face_img folder. (this feauture not yet available)  

# Makefile  
To make the command simplify, makefile was used.
```
# Normal run script
make run
# Long running script (autorestart if memory full)
make long-run
# Clean log
make clean-log
```

# Speed Improvement (CPU)
Compile tensorflow with AVX2 support  
You can built own tensorflow source code wheel file also. (take some time)  
There are online wheel file.  
[link: https://github.com/lakshayg/tensorflow-build]

# Face Detection
MTCNN  
[link: https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py]

# Face Tracking
Dlib correlation method  
[link: http://dlib.net/correlation_tracker.py.html]

# Face Recognition
AWS Rekognition / Face++  
AWS Documentation: http://boto3.readthedocs.io/en/latest/reference/services/rekognition.html  
Face++: https://console.faceplusplus.com/documents/5679127  

# Blur Image Detection
Use Laplacian Transform  

# Head Pose Estimation
DeepGaze  
[link: https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_cnn_head_pose_estimation_images/ex_cnn_head_pose_estimation_images.py]  

# Age, Gender & Emotion Estimator
Use the pretrained caffe model for agenet and gendernet  
[link: https://talhassner.github.io/home/publication/2015_CVPR]

# GUI
Tkinter(might switch to PyGtk)  

# Create Exe (Not Yet Done)
PyInstaller
