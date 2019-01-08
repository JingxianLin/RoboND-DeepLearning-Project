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

![alt text][image2]
#### 3. Parameters



#### 4. Techniques



#### 5. Image Manipulation



#### 6. Challenges



### Model

#### 1. Trained Model



#### 2. Accuracy

Here .

#### 3. Future Enhancements
