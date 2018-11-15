# tapway-ai
for all AI development codes

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
