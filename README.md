<!DOCTYPE html>
<html>

<body>
  <h1>Brain Tumor Detection Using Convolutional Neural Networks</h1>
  <p>This repository contains code for a convolutional neural network (CNN) model that detects brain tumors in MRI images. The model is built using TensorFlow/Keras and is trained on a dataset of brain MRI scans.</p>
  
  <h2>Dataset</h2>
  <p>The dataset consists of two classes:</p>
  <ul>
    <li><strong>Not_Detected</strong>: MRI images of the brain without any detected tumor.</li>
    <li><strong>Tumor_Detected</strong>: MRI images showing a detected brain tumor.</li>
  </ul>
  <p>The dataset is preprocessed and split into training and testing sets using scikit-learn's <code>train_test_split</code> function. Each image is resized to a standard size of 64x64 pixels for consistency.</p>
  
  <h2>Model Training</h2>
  <p>The model is trained using the following steps:</p>
  <ol>
    <li>Load the dataset and preprocess the images.</li>
    <li>Split the dataset into training and testing sets.</li>
    <li>Normalize the pixel values of the images.</li>
    <li>Define the CNN model architecture.</li>
    <li>Compile the model with the Adam optimizer and binary crossentropy loss.</li>
    <li>Train the model on the training data for 14 epochs.</li>
  </ol>
  
  <h2>Using the Trained Model for Prediction</h2>
  <p>To use the trained model for making predictions, follow these steps:</p>
  <ol>
    <li>Load the saved model using <code>load_model</code>.</li>
    <li>Load and preprocess the input image using <code>image.load_img</code>, <code>image.img_to_array</code>, and <code>np.expand_dims</code>.</li>
    <li>Normalize the input image pixel values to be in the range [0, 1].</li>
    <li>Get the model's prediction using <code>model.predict</code>.</li>
    <li>Display the input image with the predicted output label using <code>plt.imshow</code> and <code>plt.text</code>.</li>
  </ol>
  
  <h2>Usage</h2>
  <ol>
    <li>Clone the repository:
      <pre><code>https://github.com/anandakumarak/Brain-Tumor-Detection-Using-Convolutional-Neural-Networks.git</code></pre>
    </li>
    <li>Install the required dependencies:
      <pre><code>pip install numpy matplotlib opencv-python tensorflow scikit-learn pillow</code></pre>
    </li>
    <li>Download the pre-trained model file <code>BrainTumorDetection.h5</code> and place it in the repository directory.
    </li>
    <br>
    <li>To train your own model, follow these steps:
    </li>
    <ol>
      <li>Open the Jupyter Notebook <code>train.ipynb</code> in this repository.</li>
      <li>Run each cell in the notebook step by step to load the dataset, preprocess the images, define the CNN model architecture, compile the model, and train the model on the training data for 14 epochs.</li
    </ol>
      <h2>Saving the Trained Model</h2>
  <p>To save the trained model as a <code>.h5</code> file, use the following code in your training script:</p>
  <pre><code>model.save('BrainTumorDetection.h5')</code></pre>
  <p>This will save the entire trained model (including architecture, weights, and optimizer state) to a file named <code>BrainTumorDetection.h5</code> in the current directory.</p>
  </ol>
<li>Run the code for using the trained model for prediction:
<pre><code>python out.py</code></pre>
</li>

  
  <h2>Results</h2>
  <p>The model is able to classify brain MRI images into two classes: with tumor and without tumor. The predictions can be visualized by displaying the input image along with the predicted output label.</p>
  
  <h2>Sample Output</h2>
  <p>Below is an example of the model's prediction:</p>
  <img src = "https://github.com/A-ANANDA-KUMAR/BRAIN-TUMOR-DETECTION-USING-DEEP-LEARNING/blob/main/output.png" alt="Sample Output Image" width="900">
  
</body>
</html>

