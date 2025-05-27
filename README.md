# cnn from scratch for Handwritten Digit Classification
Introduction
This report presents the implementation of a Convolutional Neural Network (CNN) from scratch for handwritten digit classification using the MNIST dataset. The objective of the project is to understand the inner workings of CNNs by manually implementing essential components rather than relying on high-level libraries. The CNN is trained to classify handwritten digits (0-9) from the MNIST dataset, aiming to achieve high classification accuracy while using minimal external libraries.
Dataset
The MNIST dataset is used for the classification task, consisting of 60,000 training images and 10,000 testing images. Each image is a grayscale 28x28 pixel representation of handwritten digits. The images are normalized and converted to tensors for processing. The training and testing datasets are loaded using PyTorch's built-in functions.
Libraries and Tools
The following libraries and tools were used:
- NumPy: For numerical computations and matrix operations.
- PyTorch: For data loading and dataset handling.
- Matplotlib: For visualization of training progress and results.
- Scikit-learn: For calculating evaluation metrics like precision, recall, and F1-score.
Model Architecture
The CNN model is built from scratch using the following components:
1. Convolutional Layer (Conv2D): Extracts features from input images using filters/kernels. Implements forward and backward propagation with Xavier initialization for weight scaling. Supports customizable kernel size, padding, and stride.
   - Forward Method: Computes convolution by sliding the kernel over the input.
   - Backward Method: Calculates gradients of weights and biases, updating them using gradient descent.
2. Activation Functions:
   - ReLU: Applies non-linearity to activate neurons by outputting the input directly if positive, otherwise zero.
   - Softmax: Converts raw scores (logits) to probabilities that sum to one. Used in the output layer for multi-class classification.
3. Pooling Layer (MaxPool2D): Reduces spatial dimensions by selecting the maximum value from each feature map region. Uses a kernel size of 2x2 and stride of 2.
   - Forward Method: Slides the pooling window over the input, recording the max value from each region.
   - Backward Method: Propagates gradients only through the positions corresponding to the maximum values identified during the forward pass.
4. Flattening Layer: Converts the 2D pooled feature maps into a 1D vector before feeding into fully connected layers.
5. Fully Connected (FC) Layer: Applies linear transformations to the flattened input for final classification, using matrix multiplication and bias addition.
6. Loss Function (Cross-Entropy): Calculates the difference between predicted and actual labels by computing the negative log probability of the true label.
Training Process
The training process involves:
- Initializing the model with random weights.
- Forward propagation through the convolutional, pooling, and fully connected layers.
- Calculating loss using cross-entropy.
- Backpropagating errors to update weights using gradient descent.
- Iterating over the dataset for multiple epochs to minimize loss.
Results
The model's performance is evaluated using precision, recall, F1-score, and accuracy metrics. The training accuracy gradually increases as the model learns from the data, and the testing accuracy demonstrates the model's generalization ability. Visualization of loss and accuracy trends shows effective training progression.
Conclusion
The project successfully demonstrates the manual implementation of a CNN for handwritten digit classification. The results show that the custom CNN can achieve high accuracy, comparable to standard frameworks, thereby enhancing understanding of CNN operations and deep learning principles.

