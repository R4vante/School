function [x,y]=generate_dataset(N,noiseSTD)
%This function creates a dataset [x,y] for educational purposes. The 
%function 2-sin(x)-0.1x is sampled with N datapoints over the domain 
%[0,4*pi]. Gaussian noise of amplitude noiseSTD is added to the datapoints.
x = linspace(0,4*pi,N);
y = 2 -sin(x) - 0.5*x + exp(x/6) + noiseSTD*randn(size(x));

