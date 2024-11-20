# Deep Dream Image Generation

This repository contains a Colab-friendly implementation of the Deep Dream technique using TensorFlow. Deep Dream is a neural network-based algorithm that enhances patterns in an image, resulting in surreal, dreamlike visuals by amplifying features learned by a convolutional neural network (CNN).

## Overview

In this implementation, we use the pre-trained **InceptionV3** neural network, which was trained on the **ImageNet** dataset, to perform Deep Dream on an input image. By modifying parameters like the learning rate and the number of epochs, you can control how strongly the image is transformed.

### Key Functions

1. **`run_deep_dream(network, image, epochs, learning_rate)`**  
   This function performs the Deep Dream process on a given image using a pre-trained neural network. It will:
   - Modify the image to enhance features learned by the network.
   - Display the modified image every 200 epochs.
   - Return the final dreamlike image.

2. **`calculate_loss(image, network)`**  
   Calculates the loss by summing up the activations from multiple layers in the neural network. This loss is used to adjust the image during the Deep Dream process.

3. **`deep_dream(network, image, learning_rate)`**  
   Performs one step of the Deep Dream process by calculating gradients based on the loss and updating the image.

4. **`inverse_transform(image)`**  
   Converts the transformed image back into a displayable format (converting pixel values back to the correct range for visualization).

## Usage in Google Colab

1. Open the Colab notebook `Deep_dream.ipynb`.

2. Upload your input image using Colab's file upload feature or link it from a URL.

3. Modify the parameters:
   - **`epochs`**: The number of iterations to run the Deep Dream process. More epochs result in more dramatic transformations.
   - **`learning_rate`**: Controls how much the image changes during each iteration. Larger values lead to stronger modifications.

4. Run the cells to see the transformations. The image will be displayed periodically every 200 epochs, and the final result will be shown at the end.

## Example Code

To run the Deep Dream process on your image:


### Load a pre-trained model (e.g., InceptionV3)
`model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)`

### Load and preprocess an input image
`image = load_and_preprocess_image("path/to/image.jpg")`

### Run Deep Dream
`result_image = run_deep_dream(network=model, image=image, epochs=1000, learning_rate=0.01)`

## What Happens:

- The InceptionV3 network, pre-trained on the ImageNet dataset, is used to enhance specific patterns in the input image.
- After each iteration (epoch), the image is modified based on the activations in various layers of the network.
- The modified image will be displayed periodically, showing the progression of the dreamlike transformation.
- Once the specified number of epochs is completed, the final transformed image will be shown.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
