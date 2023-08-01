# Neural Network-based Character Recognition

# Datasets <br>
EMNIST:
The Extended MNIST or EMNIST dataset expands on the MNIST database commonly used as a benchmark, adding handwritten letters as well as additional samples of handwritten digits.
There are several “splits” of the data by various characteristics. We will be using the “EMNIST Letters” dataset, which contains values split into 27 classes, one unused (class 0) and one for each letter in the English alphabet.
Note: Some classes in this dataset can be challenging to recognize because each class contains images of both upper- and lower-case letters. For example, while ‘C’ and ‘c’ are very similar in appearance, ‘A’ and ‘a’ are quite different.
The file emnist_letter.npz contains EMNIST Letters in a format that can be opened with the numpy.load() method. The data contains six arrays: 'train_images', 'train_labels', 'validate_images', 'validate_labels', 'test_images', and 'test_labels'. The values have been adjusted from the original EMNIST dataset in order to match the MNIST examples included with Keras:
●	The images have been transposed and scaled to floating point.
●	The labels have been one-hot encoded.
Binary AlphaDigits:
The Binary Alphadigits dataset contains another set of handwritten letters and digits, in a different image size, in bitmap format.
The file binaryalphadigits.npz contains the letters from this dataset, in a format that can be opened with the numpy.load() method. The data contains two arrays: 'images' and 'labels'. The values have been adjusted from the original Binary Alphadigits dataset in order to match the EMNIST Letters:
●	The images of digits have been omitted.
●	The labels are in the same format as EMNIST Letters, including the unused class 0.
Note, however, that the resolution of the images is different in this dataset: 20×16 rather than 28×28.

# Tasks
# Part 1 - Warm-Up
1.	Open this notebook by Francois Chollet, which creates a simple Multilayer Perceptron as described in Section 2.1 of Deep Learning with Python, Second Edition. (Recall that this book is available from the library’s O’Reilly database.)
Chollet’s example uses the simpler MNIST dataset, which includes only handwritten digits. That dataset is included with Keras.
Run the model from this notebook. What accuracy does it achieve for MNIST?
2.	Load the EMNIST Letters dataset, and use plt.imshow() to verify that the image data has been loaded correctly and that the corresponding labels are correct.
3.	Apply the network architecture from Chollet’s MNIST notebook to the EMNIST Letters data. (You will need to modify the number of outputs, but should leave the dense layer intact.)
Note: You are welcome to use PyTorch to implement the same architecture; Keras is required only to run step (1).
What accuracy do you achieve? How does this compare with the accuracy for MNIST?
4.	The Keras examples include a Simple MNIST convnet. Note the accuracy obtained by that code compared to the previous example from Chollet.
Apply the same architecture to the EMNIST Letters data. (Again, you are welcome to implement an equivalent architecture in PyTorch instead). What accuracy do you achieve? How does this compare with the accuracy for the MNIST? How does it compare with the accuracy for EMNIST that you saw with a Dense network in step (3)?
# Part 2 - Main Event
5.	You should have found that while the EMNIST Letters are harder to learn than the MNIST digits, switching to a different network architecture led to a significant increase in model performance.
You may have noticed, however, that the training process was slower. This means that experiments take longer, and mistakes can be costly. While plotting a learning curve when training has finished can help diagnose problems, ideally you will want to see updates during the training process.
In order to avoid dead-ends while adjusting and tuning your model, TensorFlow includes the TensorBoard tool and the TensorBoard notebook extension for this purpose. While the examples show Keras models, PyTorch supports TensorBoard as well.
Add TensorBoard support to the CNN model you run in Part 1, and add the TensorBoard extension to your notebook to visualize the training process.
Note: if you get a 403 error when trying to use TensorBoard in Google Colab, you may need to enable third-party cookies.
6.	Now that you have a baseline convolutional network for comparison, begin experimenting with alternative architectures (e.g. adding additional filters to learn features and additional hidden layers to learn combinations of features) and with adjusting hyperparameters.
Your team’s goal is to obtain as high an accuracy as possible on the validation set.
Used Techniques are: 
●	Weight initialization
●	Choice of activation function
●	Choice of optimizer
●	Batch normalization
●	Data augmentation
●	Regularization
●	Dropout
●	Early Stopping
●	Pooling 

   
# Part 3 - Transfer Learning

9.	The process of transfer learning can be used to apply an existing model to a new dataset. See Transfer learning & fine-tuning in the Keras Developer Guide or the Transfer Learning for Computer Vision Tutorial in the PyTorch Tutorials.
The images in the Binary Alphadigits dataset are a different size from those in EMNIST Letters. Use a function like tf.image.resize_with_pad(), PIL.ImageOps.pad(), or the PyTorch torchvision.transforms.Resize class to resize them into the right format for the network you trained in Part 2.
10.	Is the model you trained in Part 2 capable of recognizing letters from this new dataset?
11.	Can you improve the performance on this dataset by adding additional trainable layers and fine-tuning the network?
12.	Compare the performance of the model you built in step (3) with the performance of a brand-new model trained only on the Binary AlphaDigits dataset.
Note: the dataset is so small that this may be a valid use-case for cross-validation.
What do you conclude about the value of reusing a pre-trained model?


