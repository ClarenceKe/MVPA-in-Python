# Implementation of Multi-variate Pattern Analysis Methods in Python

The human brain can easily perceive the type of objects, people's faces, facial expressions, what people are doing, as well as the social relationships between people in an image regardless of what point of view it has been taken and how complex its visual properties are. However, it is not as easy to do the same with machine vision models and algorithms, and many of them are still not precise enough in solving such problems. On the other hand, in the real world, it is very likely to face such types of images, which indicates the need to enhance the performance of the machine vision algorithms. 
In this work, we implemented some of the Multivariate Pattern Analysis (MVPA) methods in Python to study the functions of the brain in vision tasks. Using these methods could help us to understand how the brain works and perceive and how much information is available on EEG or MEG signals.

The written code mainly includes these sections:

**Pre-processing**

MEG signals are usually contaminated by different types of noise, such as magnetic and thermal ones. To reduce these noises and also to decrease the dependency of signals to the subject’s conditions, we need to filter out them. To that end, we first normalized each MEG channel by subtracting the dc value of its first 100ms part from the whole 1300ms window. Afterward, we applied a low-pass Butterworth filter with a cut-off frequency of 20 Hz. The filter was implemented using the forward-backward method to keep group delay zero.

You can change the filter parameters in the code.

**Time decoding analysis**

By plotting decoding time series, we can study at each time, how much the brain percepts, and how much the MEG patterns could show this perception.A linear SVM classifier is used in this code.

**Time-time decoding analysis**

This analysis shows how the dynamics of perception change over time, and whether patterns at one time are similar enough to classify objects at another time above the chance or not.


## Usage
You can use the dataset provided by Cichy et al. here:

[Resolving object recognition in space and time](http://userpage.fu-berlin.de/rmcichy/nn_project_page/main.html)

Please download the MEG1_MEG_Epoched_Raw_Data MATLAB matrix files from the above URL and copy them into "data". Then, run "main.py".

## Results
The results achieved for the MEG data of a subject is demonstrated here.

A sample of filtering a noisy MEG channel:

<img src="/results/sample_filtering_01_01.png" width="500" height="300">

The decoding time series plot. The green line shows the location of the achieved maximum accuracy and the blue one shows the ending of displaying image to individuals which is 500ms here. The SVM classifier accuracy signal is shown in red, however, for better visualization, a moving average version of the signal with a window size of 50, is plotted too in gray:

<img src="/results/decoding_time_series_01_01.png" width="500" height="300">

The time-time decoding plot. a) The main plot. b) A smoothed version of (a) which is obtained by applying a mean filter with a size of 50 to decrease noisy patterns:

<img src="/results/temporal_generalization_01_01.png" width="500" height="300">


## Contributing
Pull requests are welcome. If you have any question, please send me an email:

Navid Hasanzadeh: [hasanzadeh.navid@gmail.com](mailto:hasanzadeh.navid@gmail.com)

## Reference

For more information about the methods and the dataset, read this paper:
[Resolving object recognition in space and time paper by Cichy et al.](http://userpage.fu-berlin.de/rmcichy/nn_project_page/main.html)

