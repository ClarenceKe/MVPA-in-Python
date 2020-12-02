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
A sample of filtering a noisy MEG channel:

![pre-processing output](/results/sample_filtering_01_01.png =250x250)

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
