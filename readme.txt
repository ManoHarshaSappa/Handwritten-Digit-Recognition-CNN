Handwritten Digit Recognition using Convolutional Neural Networks (CNN) 

Overview 

This project focuses on building and training a Convolutional Neural Network (CNN) model to accurately classify handwritten digits from the MNIST dataset. The model is designed to handle various challenges associated with handwritten digit recognition, such as variations in writing style, digit thickness, orientation, and noise. It aims to achieve high accuracy while ensuring robust generalization on unseen data. 

Project Contributors 

Saivarun T.R. (G01475545) 

Mohammed Tareq (G01478697) 

Suraj Poldas (G01483726) 

Mano Harsha Sappa (G01459796) 


Affiliation: George Mason University 
 Course: AIT 736 - Applied Machine Learning (Spring 2025) 
 Under Guidance of: Dr. Lei Yang 


Problem Statement 

Recognizing handwritten digits accurately is a challenging task due to: 

Variations in handwriting style 

Differences in digit size, thickness, orientation, and noise 

Difficulty in capturing complex patterns in raw pixel data with traditional machine learning models 

This project aims to address these challenges by leveraging CNNs, which automatically learn hierarchical features directly from image data, providing a robust solution for handwritten digit recognition. 


Dataset Description 

Dataset Name: MNIST Handwritten Digit Database 
 Training Samples: 60,000 images 
 Testing Samples: 10,000 images 
 Classes: 10 (digits 0 to 9) 
 Image Size: 28x28 pixels, grayscale 
 Data Format: Each image is represented as a 28x28 array with pixel values ranging from 0 to 255. 


Key Challenges Addressed 

Variations in handwriting style 

Class imbalance (minor differences) 

Real-world noise and distortions 


Model Architecture 

The CNN model was designed to effectively capture spatial hierarchies in image data through layers of convolution, activation, and pooling. Key layers include: 

Convolutional Layers: Extract spatial features through learned filters 

Activation Layers: Apply non-linear transformations (ReLU) 

Pooling Layers: Reduce dimensionality and retain key features 

Fully Connected Layers: Perform final classification based on extracted features 

Training Strategy and Hyperparameter Tuning 

Training Details: 

Loss Function: Categorical Crossentropy 

Optimizer: Adam (Adaptive Moment Estimation) and SGD (Stochastic Gradient Descent) 

Evaluation Metric: Accuracy 

Batch Size: 32 

Epochs: 15 

Learning Rate: Default (0.001 for Adam) 

Early Observations: 

Rapid convergence within the first 5 epochs 

Consistently high validation accuracy (~99%) with the Adam optimizer 

 

Model Evaluation and Results 

Adam Optimizer: 

Test Accuracy: 99% 

Strong generalization with minimal overfitting 

SGD Optimizer: 

Test Accuracy: 97% 

Slower convergence but comparable final accuracy 

 

Conclusion and Future Work 

Conclusion: 

Successfully built and trained a CNN model to classify handwritten digits from the MNIST dataset. 

Achieved 99% test accuracy using the Adam optimizer, confirming the effectiveness of CNNs for this task. 

Strong model generalization with minimal overfitting, as evidenced by accuracy and loss curves. 

Future Work: 

Experiment with deeper architectures (e.g., adding more Conv layers) 

Implement regularization techniques like Dropout or L2 Regularization 

Introduce data augmentation for improved generalization 

Extend the model to handle multi-digit recognition or custom handwritten datasets 

 

Acknowledgments 

Special thanks to Dr. Lei Yang for guidance and support throughout the project. 

 

License 

This project is developed as part of the AIT 736 - Applied Machine Learning course at George Mason University. The code and results are intended for educational and research purposes only. 

 

Contact 

For any questions or collaboration, please reach out to the project contributors through their respective university email addresses. 

 