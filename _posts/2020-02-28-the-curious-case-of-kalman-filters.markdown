---
layout: post
title: "The Curious Case of Kalman Filters"  
author: krunal kshirsagar
---

Kalman filter finds the most optimum averaging factor for each consequent state. Also somehow remembers a little bit about the past states. It performs a joint probability distribution over the variables for each timeframe. The algorithm uses the new mean and new variance for every step in order to calculate the uncertainty and tries to provide accurate measurements for each timeframe of the measurement update (sensing/prediction) and the motion update(moving). The algorithm also uses other inaccuracies and statistical noise in order to represent the initial uncertainty.

# Purpose of Kalman Filters:

- **Transform data input from various sensors like LiDAR and Radar trackers into a usable form.**  
- **To Calculate inferring velocity.**  
- **Reduce measurement error(noise) of the target’s position and velocity.**  
- **Predict the future state of the target using previous state estimate and new data.**  
- **Lightweight, Robust and expandable algorithm.**  
- **Estimates a continuous state and as a result, Kalman Filters happens to give us an Uni-modal distribution.**  

# Working of Kalman Filters:

A Kalman filter gives us a mathematical way to infer velocity from only a set of measured locations. So, here I’m going to create a 1D Kalman filter that takes in positions, takes into account uncertainty, and estimates where future locations might be and the velocity of an object. Further, if we want to understand how Kalman filter works, we first need to know a bit about Gaussians which represents the Uni-modal distribution in the Kalman filters.


# Gaussian:

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-graph-representing-gaussian.png
">

**_Gaussian is a continuous function over the space of location and the area underneath sums up to 1._**  
The Gaussian is characterized by two parameters, the mean, often abbreviated with the Greek letter Mu**_( μ )_**, and the width of the Gaussian often called the variance i.e. Sigma square**_( σ² )_**. So, our job in common phases is to maintain a Mu(μ) and a Sigma square**_( σ² )_** as our best estimate of the location of the object we are trying to find. Also, remember that the larger the width, the more uncertainty it possesses.

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-the-equation-for-1d-gaussian.png
">

The diagram in the above image represents the mean**_( μ )_** and variance**_( σ² )_** of the gaussian. The taller the mean**_( μ )_**, the chances of the object present at that position are higher. conversely, if the variance**_( σ² )_** is larger, i.e. wider the distribution, higher the uncertainty of that object; that might be positioned at any place within the gaussian. And as far as the formula is concerned, it is an exponential of a quadratic function where we take the exponent of the expression. The quadratic difference of our query point x, relative to the mean**_( μ )_**, divided by sigma square**_( σ² )_**, multiplied by -(1/2). Now if **_x_** = **_μ_**, then the numerator becomes 0, and if x of 0, which is 1. It turns out we have to normalize this by a constant, 1 over the square root of 2 Pi**_(π)_** sigma square**_( σ² )_**.

# Gaussian Characteristics:

Gaussians are exponential function characterized by a given mean**_( μ )_**, which defines the location of the peak of a Gaussian curve, and a variance**_( σ² )_** which defines the width/spread of the curve. All Gaussian are:  
- **symmetrical**  
- they have **one peak**, which is also referred to as a “unimodal” distribution, and they have an exponential drop off on either side of that peak.

# More on Variance:

The variance is a measure of Gaussian spread; larger variances correspond to shorter Gaussians. Variance is also a measure of certainty; if you are trying to find something like the location of a car with the most certainty, you’ll want a Gaussian whose mean is the location of the car and with the smallest uncertainty/spread.

# Let’s write a Gaussian function:  
```python
from math import *
import matplotlib.pyplot as plt
import numpy as np

# gaussian function
def f(mu, sigma2, x):
    ''' f takes in a mean and squared variance, and an input x
       and returns the gaussian value.'''
    coefficient = 1.0 / sqrt(2.0 * pi *sigma2)
    exponential = exp(-0.5 * (x-mu) ** 2 / sigma2)
    return coefficient * exponential
```
# Shifting the Mean:

In Kalman filters, we iterate through measurement (measurement update) which uses Bayes rule, which is nothing else but a product or multiplication and through motion update(prediction) in which we use total probability which is a convolution or simply an addition.

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-mysterious-cycle.png
">

In order to understand the cycle, let’s assume that we are localizing a vehicle and we have a prior distribution(blue gaussian); it is a very wide Gaussian with the mean. and now let’s say we get a measurement(orange gaussian) that tells us something about the localization of vehicle. This is an example in our prior we were fairly uncertain about the location but the measurement told us quite a bit as to where the vehicle is.

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-prior-distribution-measurement-distribution.png
">

>Note: In the above diagram, Mu**_( μ )_** is the prior mean and Nu**_( v )_** is the new measurement mean.

The final mean gets shifted which is in between the two old means, the mean of the prior, and the mean of the measurement. It’s slightly further on the measurement side because the measurement was more certain as to where the vehicle is than prior. The more certain we are, the more we pull the mean on the direction of the certain answer.

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-new-mean.png
">

# Where is the new peak?  

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-new-peak.png
">

The resulting Gaussian is more certain than the two-component Gaussians i.e. the covariance is smaller than either of the two covariances in the installation. Intuitively speaking, this is the case because we actually gain information. The two Gaussians together with high information content in either Gaussian installation.


# Parameter Update:

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-measurement-update.png
">

1. Suppose we multiply two Gaussians, as in Bayes rule, a prior and a measurement probability. The prior has a mean of Mu**_( μ )_** and a variance of Sigma square**_( σ² )_**, and the measurement has a mean of Nu**_( v )_** and covariance of r-square**_( r² )_**.  
2. Then, the new mean, Mu prime**_( μ′ )_**, is the weighted sum of the old means. The Mu**_( μ )_** is weighted by r-square**_( r² )_**, Nu**_( v )_** is weighted by Sigma square**_( σ² )_**, normalized by the sum of the weighting factors. The new variance term would be Sigma square prime**_( σ²′ )_**.  
3. Clearly, the prior Gaussian has a much higher uncertainty, therefore, Sigma square**_( σ² )_** is larger and that means the Nu**_( v )_** is weighted much much larger than the Mu**_( μ )_**. So, the mean will be closer to the Nu**_( v )_**than the Mu**_( μ )_**. Interestingly enough, the variance term is unaffected by the actual means, it just uses the previous variances.

```python
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean, new_var]
```

# Gaussian Motion:

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-motion-update.png
">

A new Mean**_( μ′ )_** is your old Mean Mu**_( μ )_** plus the motion often called **u**. So, if you move over 10 meters in the x-direction, this will be 10 meters and you knew Sigma square prime**_( σ²′ )_** is your old Sigma squared**_( σ² )_** plus the variance**_( r² )_** of the motion Gaussian. This is all you need to know, it’s just an addition. The resulting Gaussian in the prediction step just adds these two things up, mu**_( μ )_** plus **u** and sigma squared**_( σ² )_** plus **_( r² )_**.

```python
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]
```


# The Filter Pipeline:

- So now let’s put everything together. Let’s write the main program that takes these 2 functions, update and predict, and feeds into a sequence of measurements and motions. In the example I’ve chosen, measurements = 5., 6., 7., 9., 10. and motions are 1., 1., 2., 1., 1. This all would work out really well if the initial estimate was 5, but we’re setting it to 0 with a very large uncertainty of 10,000.  
- Let’s assume the measurement uncertainty is constant 4, and the motion uncertainty is constant 2. When you run this, your first estimate for the position should basically become 5–4.99, and the reason is your initial uncertainty is so large, the estimate is dominated by the first measurement. Your uncertainty shrinks to 3.99, which is slightly better than the measurement uncertainty. You then predict that you add 1, but the uncertainty increases to 5.99, which is the motion uncertainty of 2.  
- You update again based on measurement 6, you get your estimate of 5.99, which is almost 6. You move 1 again. You measure 7. You move 2. You measure 9. You move 1. You measure 10, and you move a final 1. And outcomes as the final result, a prediction of 10.99 for the position, which is your 10 position moved by 1, and the uncertainty(residual uncertainty) of 4.

```python
# measurements for mu and motions, U
measurements = [5., 6., 7., 9., 10.]
motions = [1., 1., 2., 1., 1.]

# initial parameters
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.  #0000000001  

## TODO: Loop through all measurements/motions
## Print out and display the resulting Gaussian 

# your code here
for i in range(len(measurements)):
    # measurement update, with uncertainty
    mu, sig = update(mu, sig, measurements[i], measurement_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    # motion update, with uncertainty
    mu, sig = predict(mu, sig, motions[i], motion_sig)
    print('Predict: [{}, {}]'.format(mu, sig))

    
# print the final, resultant mu, sig
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))
```
# Output  
#### When the initial uncertainty is high:  

<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-when-the-initial-uncertainty-is-high.png
">

# Plotting the Gaussian:

```python
## Print out and display the final, resulting Gaussian 
# set the parameters equal to the output of the Kalman filter result
mu = mu
sigma2 = sig

# define a range of x values
x_axis = np.arange(-20, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)
```
<img src="{{ site.baseurl }}/images/2020-02-28-the-curious-case-of-kalman-filters-resulting-gaussian.png
">

# How the Filter actually works:
>Remember that measurement_sig(var2 or σ²) = 4 in measurement update & motion_sig(var2 or σ²) = 2 in motion update(prediction) will remain constant. Other than that each and every variable in the code will be updated at each timestep. for example, in the for loop of the filter pipeline, the mu and sig values will get updated in the measurement update and the new updated mu and sig values are then fed into the motion update(predict) function. then again when the new values are generated by the motion update(predict) function, those updated values are then fed into the measurement update(update) function and the cycle goes on till it calculates uncertainty for each and every timestep of the vehicle/object.
Feel free to mess around with the code and understand what is really happening when you change a particular variable’s value. **Also, change the initial uncertainty to a very low value such as 0.0000000001 in order to get a clear understanding of How the Kalman Filter works!**

Check out the code on **[Github](https://github.com/Noob-can-Compile/Kalman_Filters)**.

