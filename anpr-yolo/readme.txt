This project is an attempt to train a YOLO model for car plate recognition and detection.

### What the the folders and files?

Folders
-------

cameras-metas
- XML generated from labelImg
- note: original CCTV frames are NOT included in this project due to size.

cameras-outputs
 - Images of cars with carplate.
 - JSON files that contains the bounds of carplate, and each characters in the carplate.
 - JSON is parsed from XML in cameras-metas.
 - The main_dev_samples() in script [version]_image.py generate the JSON from XML.

cameras-trains
- Samples output from Python scripts.
- This is to check if generated synthetic data is correct.

models
- Folders to save trained models.

outputs
- Folders to save test images result from trained models

outputs-sampled
- Some selected sampled to use with notes.txt document.


Files
-----

* NOTE: aero_, bisque_, etc ... are just version code name for the scripts.

aero_image.py = Generate synthetic training and dev images. (SIMPLE)
aero_yolo.py  = YOLO implementation

bisque_image.py = Generate synthetic training and dev images. (Diff brightness, rotations, sizes)
bisque_yolo.py  = YOLO implementation

cadet_image.py = Generate synthetic training and dev images.
cadet_yolo.py  = YOLO implementation. Test on REAL carplate images.

denim_image.py = Generate synthetic training and dev images. (Training data with REAL background)
denim_yolo.py  = YOLO implementation. Test on REAL carplate images.

earth_image.py = Generate synthetic training and dev images. (Training data with REAL background and black character background)
earth_yolo.py  = YOLO implementation. Test on REAL carplate images.


- The samples output can be found at outputs-sampled.
- The results are presented in notes.txt.


