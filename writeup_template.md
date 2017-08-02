#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-model.png "Model Visualization"
[image2]: ./examples/center_driving.jpg "Center Lane Driving"

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model first normalizes the data using a keras lambda layer (model.py:71). I then crop out irrelevant image data (model.py:72), including the sky and hood of the car. 

The next five layers are 2D convolutional layers. The filter dimensions and kernel sizes match those noted in the nvidia paper (model.py:73-77). RELUs are used as activation
functions in each convolutional layer.   

####2. Attempts to reduce overfitting in the model

I did not utilize dropout, as I did not observe loss patterns that indicated overfitting (that is, I did not see training loss fall fast while validation loss remained high).
 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py:84).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I created the data by controlling the cars input with a mouse to generate better steering angles.
I used a combination of center lane driving in the forward and reverse directions, as well as a number of recovery drives from the side of the track back to the center. I also
augmented the data by flipping the images over the Y-axis and inverting the steering angles. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first start with a very basic model (a flattened image through a single node, as shown in lecture) to ensure that all
pieces of the pipeline were working. Then I read through the NVIDIA whitepaper and decided to implement their model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, namely the road directly after the bridge. To improve the driving behavior in these cases, I 
recorded a few more laps of track data, and also recorded some more recovery driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py:69-84) consisted of a convolution neural network that matched what can be found in the NVIDIA whitepaper.

Here is a visualization of the NVIDIA CNN architecture:

![The NVIDIA CNN Architecture][image1]

You can read more about it here: [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to the center of the lane.

To augment the data sat, I also flipped images and angles thinking that this would help reduce the bias toward left-hand turns that the first course introduces (by virtue of being circular).

After the collection process, I had about 60,0000 data points. I then preprocessed this data by normalizing it and cropping out details that did not impact the model (sky and car details).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs seemed to be around 3 (judging by the loss graph that was created after each model training). Ultimately I used 5 epochs and didn't see any detrimental impacts (overfitting).  I used an adam optimizer so that manually training the learning rate wasn't necessary.
