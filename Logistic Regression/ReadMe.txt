{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf470
{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red58\green62\blue68;\red237\green236\blue236;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl300\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Nearest Neighbor\
It absolutely doesn't make sense that someone who studies 1.75 hours has a higher likelihood of passing than some who studied 2.00 hours. You don't know if your hitting noise or signal. We need to look at the structure of the data. \
\
K nearest neighbors\
This is better than the nearest neighbor approach, but is it reliable? Not very. This method might be better at overcoming noise vs signal, but for k=3 the probability can only be 0%, 33%, 67% and 100%. \
\
Model based\
Objective function part 1\
How were these values picked: 1 hour or less and the chances of parssing are one in 128 and those chances double every 45 minutes? It's very arbitrary. We have  no idea if this data fits an exponential curve.\
\
Parameterized model\
I like that we are finding the best fit for a type of model now, but again, we have no idea if the data fits an exponential curve. I don't like that we are saying the probability of passing doubles every x minutes and we are still picking an arbitrary starting point. And do we jump from one out of 2 chances of passing to 1 out of 1 chances of passing in x minutes? That's a pretty rapid jump over x minutes. I think the curve should be more asymptotic.\
\
Logistic Regression\
I'm much happier now. We are fitting two parameters and fitting  a more appropriate curve. I believe this data should exhibit a "learning curve"; there is a certain amount of time of studying that does not improve chances of passing, a slow beginning. Then there might be a steep acceleration leading to a plateau that has a much higher chance of passing (but never 100%).}