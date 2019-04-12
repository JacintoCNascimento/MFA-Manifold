
clc; clear all; close all;
randn('state',0); rand('state',0);
load Face_data.mat

[p0,n0] = size(images);
ord = randperm(n0);

X = images(:,ord(1:(n0-50)));  
label = [lights(ord(1:(n0-50)))' poses(:,ord(1:(n0-50)))'];
[p,n] = size(X); 
k = 50*ones(50,1); 

para.k = k; para.cet = 1;
para.maxit = 2500; para.num = 500;
spl = MFA_DP(X,para);

%save('Face_Result.mat', 'spl');
