## Project: Follow Me
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


[//]: # (Image References)

[image1]: ./figures/searched.png
[image2]: ./docs/misc/fcn.png
[image3]: ./figures/threshed.png 

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

![alt text][image2]
#### 3. Hyperparameters
```python
learning_rate = 0.01
batch_size = 16
num_epochs = 50
steps_per_epoch = 4131 / batch_size
validation_steps = 100
workers = 2
```

And this is the hardest part of this project.  
#### 4. Techniques



#### 5. Image Manipulation



#### 6. Challenges



### Model

#### 1. Trained Model



#### 2. Accuracy

Here .

#### 3. Future Enhancements
