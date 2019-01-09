## Project: Follow Me
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


[//]: # (Image References)

[image1]: ./docs/misc/model.png
[image2]: ./docs/misc/fcn.png
[image3]: ./docs/misc/following1.png
[image4]: ./docs/misc/following2.png
[image5]: ./docs/misc/far1.png
[image6]: ./docs/misc/far2.png
[image7]: ./docs/misc/non1.png
[image8]: ./docs/misc/non2.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

#### 2. Architecture
The final project is based on the Semantic Segmentation lab and trying to build a deep learning model that will allow a simulated quadcoper to follow around a particular human target it detects.  In order to fulfill this task, a Fully Convolutional Network (FCN) is trained to label each pixel in the output image as one of environment, person, and hero, so the drone can follow the hero closely and accurately.  FCN can preserve spatial information throughout the network, but CNN cann't.

An FCN is comprised of an encoder and decoder: The encoder portion is a convolution network that reduces to a deeper 1x1 convolution layer, in contrast to a flat fully connected layer used for basic classification of images; the decoder section is made of reversed convolution layers.

Encoder includes a separable convolution layer, with the code:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

1x1 convolution layer is a regular convolution, its code followed:

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

Decoder is comprised of three parts: A bilinear upsampling layer, a layer concatenation step, and a separable convolution layer:

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    small_ip_layer_upsampled = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([small_ip_layer_upsampled, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(input_layer = output_layer, filters = filters)
    return output_layer
```

Final model is 7 layers deep, with 3 encoder blocks, 1x1 convolution layer, and 3 decoder blocks:

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    conv1 = encoder_block(input_layer = inputs, filters = 32, strides = 2)
    print("conv1: ", conv1.shape)
    conv2 = encoder_block(input_layer = conv1, filters = 64, strides = 2)
    print("conv2: ", conv2.shape)
    conv3 = encoder_block(input_layer = conv2, filters = 128, strides = 2)
    print("conv3: ", conv3.shape)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_norm = conv2d_batchnorm(input_layer = conv3, filters = 128, kernel_size = 1, strides = 1)
    print("conv_norm: ", conv_norm.shape)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    deconv1 = decoder_block(small_ip_layer = conv_norm, large_ip_layer = conv2, filters = 64)
    print("deconv1: ", deconv1.shape)
    deconv2 = decoder_block(small_ip_layer = deconv1, large_ip_layer = conv1, filters = 32)
    print("deconv2: ", deconv2.shape)
    deconv3 = decoder_block(small_ip_layer = deconv2, large_ip_layer = inputs, filters = num_classes)
    print("deconv3: ", deconv3.shape)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    output_layer = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(deconv3)
    print("output_layer: ", output_layer.shape)
    return output_layer

conv1:  (?, 80, 80, 32)
conv2:  (?, 40, 40, 64)
conv3:  (?, 20, 20, 128)
conv_norm:  (?, 20, 20, 128)
deconv1:  (?, 40, 40, 64)
deconv2:  (?, 80, 80, 32)
deconv3:  (?, 160, 160, 3)
output_layer:  (?, 160, 160, 3)
```

![alt_text][image2]
#### 3. Hyperparameters
```python
learning_rate = 0.01
batch_size = 16
num_epochs = 50
steps_per_epoch = 4131 / batch_size
validation_steps = 100
workers = 2
```

And this is the hardest part of this project.  For learning rate, 0.1 is too large, making poor performance; 0.01 brings down the loss, and achieves an accuracy greater than 0.4.  Batch size of 16 is good for memory and training speed.  Number of epochs is set to 50, because of low training and validation loss; fewer epochs tend to underfit the model with high training and validation loss; more epochs are prone to overfitting with low training loss and high validation loss.  Steps per epoch is based on the total number of images in training dataset divided by the batch_size.  For validation steps and workers, recommended values are used.
#### 4. Model Results

Here are some results after testing the trained model.  Image on the left is the raw one, the middle is the ground truth, and the right is model output.  While the quad is following behind the target, average IoU for the hero is 0.901, pretty high, shown below.

![alt_text][image3]
![alt_text][image4]

While the quad is on patrol and the target is not visible, average IoU for other people is 0.729, examples here:

![alt_text][image7]
![alt_text][image8]

While the target is far away, average IoU for the hero is 0.229, not so good, these hard cases as follows:

![alt_text][image5]
![alt_text][image6]

Trained model weights are saved in the correct format and run without errors, and several different scores to evaluate the model are included:

```python
# Sum all the true positives, etc from the three datasets to get a weight for the score
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
print(weight)
0.7396514161220044
# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU = (iou1 + iou3)/2
print(final_IoU)
0.565085841466
# And the final grade score is 
final_score = final_IoU * weight
print(final_score)
0.417966542871
```

This model and data would not work well 

#### 5. Future Enhancements
