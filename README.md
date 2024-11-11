# Custom_CNN_based_Hand-_Gesture_Recognition
# Data Preprocessing
The dataset has 2515 images with 36 classses (0-9 to a-z) where each class has 70 images except class t having 65 images. The dataset is splitted in 80:20 ratio with 2012 images for training and 503 images for validation after augmentation with a target size of 64 X 64. For testing, all 2515 images were used without augmentation.
The dataset is loaded from a directory structure containing subfolders for each gesture class.
Image Augmentation: The model uses ImageDataGenerator for augmenting the training data with techniques like rescaling, random width/height shifts, shear transformations, and zooms. This helps improve generalization.
The data is split into training (80%) and validation (20%) sets, with separate test data loading without augmentation.
# Exploratory Data Visualization
The visualization displays 10 images along with their corresponding class labels, which helps ensure the data is loaded correctly.
# Model Architecture: The model is a Sequential CNN with the following architecture:
Convolutional Layers: 4 convolutional layers (32, 64, 128, and 256 filters) followed by max-pooling.
Batch Normalization: Applied after the final convolutional layer to help normalize activations.
Regularization: L2 regularization is applied to the fully connected layer to reduce overfitting.
Dropout: A dropout layer with a rate of 0.5 is used to further prevent overfitting.
Dense Layers: Two fully connected layers, the last one being a softmax layer with 36 units (one per class).
# Model Compilation
The model is compiled using the Adam optimizer with a learning rate of 0.001, and categorical cross-entropy as the loss function.
The accuracy metric is also tracked during training.
# Training and test results
The model was trained for 25 epochs (out of a possible 50) as early stooping was used to avoid overfitting.
The final training accuracy reached around 88.07%, with the validation accuracy at 86.88% by Epoch 25.
The test accuracy was 94.00%, indicating strong generalization to the test data.
The test loss was 0.5093, which is relatively low, supporting the good test accuracy.
# Observation from training and validation graph
In accuarcy plot, The model's accuracy improves steadily for both training and validation sets, stabilizing around 80% to 90% by the end.
In loss plot, both losses decrease sharply in the first few epochs and then flatten out, indicating effective learning. The validation loss follows the same trend as the training loss, but the two lines converge well after epoch 10, indicating that the model generalizes well without severe overfitting.
# Observation from Classification report and Confusion Matrix
Overall Accuracy: 94% (on the test set with 2,515 samples).
# Class-wise Analysis:
High Performance:
Several classes like 3, 7, 8, 9, and others show perfect precision, recall, and F1-scores of 1.00, meaning the model perfectly predicted those classes.
Classes with near-perfect performance include class f (F1-score of 0.99) and class h (F1-score of 1.00).
Challenges:
Class 6 shows some difficulty with lower recall (0.50) and F1-score (0.65), indicating that many instances of class 6 were misclassified.
Class 1 has a high precision (1.00) but a lower recall (0.66), indicating over-prediction in this class.
Class v also had challenges, with a lower F1-score of 0.68 and recall of 0.51.
# Confusion matrix analysis:
Perfect Predictions: Several classes, such as 3, 7, h, x, and others, show no errors, indicating the model predicts them with complete accuracy.
Misclassifications: Class 6 seems to have some confusion with other classes, as observed from the off-diagonal entries. The model also struggles a bit with classes v and w.
# Predictions from the dataset
Plotted 10 random images from the test data with 10 actual and predicted labels.
Accurately predicted 10 random classes for the respective images.
# Real time Gesture Recognition
Used Web CAM to capture the gesture.
Imports: cv2 for webcam handling, numpy for numerical processing.
Class Labels: Defined for digits ('0'-'9') and letters ('a'-'z').
Webcam Setup: Initializes video capture, retrieves frame dimensions, and defines a 300x300 ROI on the right side of the frame.
Pre-processing: The pre_process_frame() function resizes the ROI to 64x64 and normalizes pixel values.
Main Loop: Reads webcam frames, converts them to grayscale, applies Gaussian blur(to reduce noise), and thresholds to detect hand contours.If contours are found, it draws a blue rectangle for the ROI, extracts the ROI, preprocesses it, and prepares it for model prediction.Displays predicted class label on the screen based on the model output.Exit: Stops if 'q' is pressed.
# Challenges Encountered for real time prediction
When implementing real-time gesture prediction using an American Sign Language (ASL) dataset with a webcam inconsistent lighting significantly affected the quality of the video feed, making it difficult for gesture recognition algorithms to accurately identify some ASL signs.
The performance of ASL gesture recognition was affected by the computational power of the device running the algorithm.
# Further Recommendation
Pre trained models can be used instead of training the CNN from scratch, reducing training time and ensuring faster convergence.
Hyper parameter tuning can be done such as adjusting learning rate, optimiser, adding more dropout layers or increasing no of convolution layers.
