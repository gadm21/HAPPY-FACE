# How to setup MySQL
[link: https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-18-04]

# Create Database + Table (Only First Time)
Run the command to create database and related table
```
sudo mysql -u <username> -p < setup.sql
```

# How to run the code
1. export python path
refer to here:
https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1

2. setup aws credentials by running this command: aws configure

3. Run those command:  
pip3 install -r requirements.txt  
cd face-detect/facenet/src  
python3 tapway-face.py  

then the cropped image will save in face_img folder.

Refer here for more details:
https://github.com/davidsandberg/facenet

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
Blur Detection for Digital Images Using Wavelet Transform  
Modify code from  
[link: https://gist.github.com/dosas/4369287]  
Paper:  
[link: https://www.cs.cmu.edu/~htong/pdf/ICME04_tong.pdf]  

# Head Pose Estimation
DeepGaze  
[link: https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_cnn_head_pose_estimation_images/ex_cnn_head_pose_estimation_images.py]  

# Age & Gender Estimator
Use the pretrained caffe model for agenet and gendernet  
[link: https://talhassner.github.io/home/publication/2015_CVPR]

# GUI
Tkinter  

# Create Exe (Not Yet Done)
PyInstaller
