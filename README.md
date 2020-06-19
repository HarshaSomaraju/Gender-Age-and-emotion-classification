# Gender-Age-and-emotion-classification
A model to predict both gender and age and another model to predict emotion

Gender,age and emotion classification (from scratch) using convolution neural network

The keras model is created by training VGG based model from scratch on around 38,000 preprocessed images for gender,age and 4000 preprocessed images for emotion. 
It acheived around 95% training accuracy and ~90% validation accuracy. (20% of the dataset is used for validation) for wiki dataset.

In this repository, Manual Dataset is used for training. It is not as good as training using wiki dataset but gives considerable results.

Python packages

numpy
pandas
opencv-python
tensorflow
keras

Install the required packages by executing the following command.

$ pip install -r requirements.txt

Note: This repo works on Python 3.x

Usage

$ python all_output_program_images.py
(path should be set in the above python file 'path' variable)
(Sample images can be given for testing.They are in Sample folder)

$ python all_output_program_video.py
(This file can be used for video stream)

 Model Building

 multi_output_model.py
 > This creates the base convolution neural network models for 1. Gender,age 2. Emotion that is used to train and predict.

 Training

 flow_from_dataframe_train_multi.py
 > It is used to train gender-age model. It takes input from Manual dataset. Manual dataset contains less number of images and can be trained faster. This dataset is manually created by scrapping data from google and preprocessed.

 flow_from_directory_emotion.py
 > It is used to train emotion model. It takes input from Genki4k preprocessed dataset. Path of the dataset need to be specified. All the images are fitted and the model is trained then saved.

 Note: Base and Trained models are already provided in the project as
 
 Base models:
 1. 2-class-base-dropouts.h5 for gender and age
 2. Emotion_Base_model.h5 for emotions

 Trained models:
 1. 2-Trained_model.h5 for gender and age.
 2. trained_emotion_model.h5 for emotion.
