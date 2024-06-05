
## On Image Data Augmentation: Training CNNs with Low-Rank Approximation of Images using the Singular Value Decomposition

**Kennette Basco**  
New York Institute of Technology  
Department of Engineering and Computer Science

## SVDNet
SVDNet is a feature extracted and fine-tuned pre-trained VGG16 model using SVD-augmented images


![MNIST](/resources/mnist.gif)![PATCH](/resources/patch_camelyon.gif)![MALARIA](/resources/malaria.gif)![CATSDOGS](/resources/cats_vs_dogs.gif)

## ABSTRACT
This study investigates enhancing the generalization of Convolutional Neural Networks (CNNs) for low-resolution images using Singular Value Decomposition (SVD) augmented images. It provides a theoretical overview of SVD and a method to control compression through hyperparameters: energy factor and skip threshold. Methods involve preprocessing, training models, and benchmarking validation and testing accuracy on various datasets (grayscale and RGB). Experiments using SVD augmentation show degraded validation and testing accuracy. In summary, while SVD augmented data shows theoretical promise, it does not alleviate the training-production data skew and even increases the signal-noise ratio. Rigorous hyperparameter tuning of the energy factors and skip threshold may regularize CNNs, but the computational resources required outweigh the marginal gains in regularization, if at all. This study highlights the limitations of SVD based augmentation and underscores the need for alternative regularization techniques in deep learning.
Keywords: convolutional neural networks, singular value decomposition, regularization, adversarial learning.


## Other Resources
- Paper: https://tinyurl.com/kbasco-svd
- Presentation: https://tinyurl.com/kbasco-svd-ppt
- SVDNet Keras Model: https://tinyurl.com/kbasco-svd-model
