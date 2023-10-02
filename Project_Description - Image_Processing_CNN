This is a guided Project from DataCamp. I have been introduced to CNN Model through this mini-project. Its a binary Classification Problem.
Can a machine identify a bee as a honey bee or a bumble bee? These bees have different behaviors and appearances, but given the variety of backgrounds, positions, and image resolutions, it can be a challenge for machines to tell them apart.
Being able to identify bee species from images is a task that ultimately would allow researchers to more quickly and effectively collect field data. Pollinating bees have critical roles in both ecology and agriculture, and diseases like colony collapse disorder threaten these species. Identifying different species of bees in the wild means that we can better understand the prevalence and growth of these important insects.

1) Load Libraries

2) Load Dataframe

3)Examine RGB values in an image matrix
For each pixel in an image, there is a value for every channel. The combination of the three values corresponds to the color, as per    the RGB color model. Values for each color can range from 0 to 255, so a purely blue pixel would show up as (0, 0, 255).

4) 4. Importing the image data
Now we will import all images. Once imported, we will stack the resulting arrays into a single matrix and assign it to X.

5) Split into train, test, and evaluation sets
Now that we have our big image data matrix, X, as well as our labels, y, we can split our data into train, test, and evaluation sets. To do this, we'll first allocate 20% of the data into our evaluation, or holdout, set. This is data that the model never sees during training and will be used to score our trained model.
We will then split the remaining data, 60/40, into train and test sets just like in supervised machine learning models. We will pass both the train and test sets into the neural network.

6) Normalize image data

Now we need to normalize our image data. Normalization is a general term that means changing the scale of our data so it is consistent.
In this case, we want each feature to have a similar range so our neural network can learn effectively across all the features. As explained in the sklearn docs, "If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected."
We will scale our data so that it has a mean of 0 and standard deviation of 1. We'll use sklearn's StandardScaler to do the math for us, which entails taking each value, subtracting the mean, and then dividing by the standard deviation. We need to do this for each color channel (i.e. each feature) individually

7) Model building (1)
It's time to start building our deep learning model, a convolutional neural network (CNN). CNNs are a specific kind of artificial neural network that is very effective for image classification because they are able to take into account the spatial coherence of the image, i.e., that pixels close to each other are often related.

Building a CNN begins with specifying the model type. In our case, we'll use a Sequential model, which is a linear stack of layers. We'll then add two convolutional layers. To understand convolutional layers, imagine a flashlight being shown over the top left corner of the image and slowly sliding across all the areas of the image, moving across the image in the same way your eyes move across words on a page. Convolutional layers pass a kernel (a sliding window) over the image and perform element-wise matrix multiplication between the kernel values and the pixel values in the image.

NOTE- n the first convolutional layer, you have 32 filters stacked to form the layer. Within each filter, there are processing units (sometimes referred to as neurons) that share the same weights and have an activation function. These processing units within a filter work together to detect different aspects or variations of the same feature in the input data (e.g., edges, textures, or other patterns).

So, to clarify:

The first convolutional layer consists of 32 filters, each responsible for detecting specific patterns or features.
Within each of these 32 filters, there are processing units, which you can think of as neurons, although they work differently from neurons in a fully connected layer.
These processing units within a filter share the same weights and apply an activation function (typically ReLU) to their inputs.

8) Model building (2)

Let's continue building our model. So far our model has two convolutional layers. However, those are not the only layers that we need to perform our task. A complete neural network architecture will have a number of other layers that are designed to play a specific role in the overall functioning of the network. Much deep learning research is about how to structure these layers into coherent systems.

We'll add the following layers:

MaxPooling. This passes a (2, 2) moving window over the image and downscales the image by outputting the maximum value within the window.
Conv2D. This adds a third convolutional layer since deeper models, i.e. models with more convolutional layers, are better able to learn features from images.
Dropout. This prevents the model from overfitting, i.e. perfectly remembering each image, by randomly setting 25% of the input units to 0 at each update during training.
Flatten. As its name suggests, this flattens the output from the convolutional part of the CNN into a one-dimensional feature vector which can be passed into the following fully connected layers.
Dense. Fully connected layer where every input is connected to every output (see image below).
Dropout. Another dropout layer to safeguard against overfitting, this time with a rate of 50%.
Dense. Final layer which calculates the probability the image is either a bumble bee or honey bee.

To take a look at how it all stacks up, we'll print the model summary. Notice that our model has a whopping 3,669,249 paramaters. These are the different weights that the model learns through training and what are used to generate predictions on a new image.

9) Compile and train model
Now that we've specified the model architecture, we will compile the model for training. For this we need to specify the loss function (what we're trying to minimize), the optimizer (how we want to go about minimizing the loss), and the metric (how we'll judge the performance of the model).
Then, we'll call .fit to begin the trainig the process.
"Neural networks are trained iteratively using optimization techniques like gradient descent. After each cycle of training, an error metric is calculated based on the difference between prediction and target…Each neuron’s coefficients (weights) are then adjusted relative to how much they contributed to the total error. This process is repeated iteratively.

10) Load pre-trained model and score

NOTE- In a machine learning model, such as a neural network, the values for weights and biases are set and updated through a process called training. Training is the phase where the model learns from the provided data to make accurate predictions. The values for weights and biases are initialized with small random values at the beginning of training, and then they are updated iteratively during training to minimize a predefined loss or cost function. Here's how this process works:

Initialization: Initially, the model's weights and biases are assigned random values, typically drawn from a small random distribution. This step is crucial because it breaks the symmetry in the model and allows it to start learning.
Forward Pass: During each training iteration (or epoch), the model takes a batch of input data and performs a forward pass. This involves the following steps:

The input data is passed through the neural network, layer by layer, using the current values of weights and biases.
Each layer performs mathematical operations on the input data, applying weights and biases as specified in the layer's architecture.
The final layer produces predictions (outputs) based on the input data and the current parameter values.
Loss Calculation: After making predictions, the model calculates a loss or error value. This loss quantifies how far off the model's predictions are from the actual target values. Common loss functions include mean squared error for regression tasks and binary cross-entropy for binary classification tasks.

Backpropagation: The core of training is the backpropagation algorithm. It computes the gradient of the loss with respect to each parameter (weights and biases) in the network. This gradient indicates how much the loss would change if each parameter were adjusted slightly.

Parameter Update: The gradients calculated in the previous step are used to update the values of weights and biases. The goal is to adjust these parameters in a way that reduces the loss. This is typically done using optimization algorithms such as Stochastic Gradient Descent (SGD) or Adam. The update rule is usually of the form:

new_weight = old_weight - learning_rate * gradient
new_bias = old_bias - learning_rate * gradient
Repeat: Steps 2 to 5 are repeated for a predefined number of epochs or until the model converges, meaning that the loss stops improving significantly.

Final Model: Once training is complete, the model's weights and biases are set to the values that minimize the loss function. These values represent the learned knowledge of the model and can be used for making predictions on new, unseen data.

10)Load pre-trained model and score
the evaluate method to see how well the model did at classifying bumble bees and honey bees for the test and validation sets. Recall that accuracy is the number of correct predictions divided by the total number of predictions. Given that our classes are balanced, a model that predicts 1.0 for every image would get an accuracy around 0.5.

Conclusion -  the model generalizes quite well as the accuracy is similar for the test set and holdout set: 0.66 for data the model has seen (the test set) and 0.65 for data the model has not seen (the holdout set).

11) Visualize model training history
In addition to scoring the final iteration of the pre-trained model as we just did, we can also see the evolution of scores throughout training thanks to the History object. We'll use the pickle library to load the model history and then plot it.
Notice how the accuracy improves over time, eventually leveling off. Correspondingly, the loss decreases over time. Plots like these can help diagnose overfitting. If we had seen an upward curve in the validation loss as times goes on (a U shape in the plot), we'd suspect that the model was starting to memorize the test set and would not generalize well to new data.

12) Generate predictions
 predicts honey bee or bumble bee

