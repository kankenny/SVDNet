
# Image Data Augmentation: Low Rank Approximation Using the Singular Value Decomposition
---
##### Kennette Basco 
##### Master's project 
##### New York Institute of Technology 
##### Department of Engineering and Computer Science 

---
Data augmentation, particularly in image processing/computer vision, is a useful regularization method in training deep convolutional neural networks (CNN). It entails introducing variations in the training data that aid the CNN model to recognize patterns and spatial properties more effectively. Common techniques include: random alterations in orientation, rotation, scale, addition of pixel value noise, etc. Another potential addition to image augmentation techniques is using a low-rank approximation of images through singular value decomposition (SVD) to compress images while retaining crucial pixel information. 
Regularization is any method that decreases the optimization performance (some metric on the training data) that improves generalization performance (some metric on the testing data). Since the goal of machine learning is to predict data that has not been seen yet, regularization of models is crucial to avoid overfitting the training data. The regularization methods on images aforementioned are only active during training, i.e., at inference time, the images are not distorted.
The regularization effect of SVD will be determined in the binary and multiclass classification settings using a trained deep learning model.


