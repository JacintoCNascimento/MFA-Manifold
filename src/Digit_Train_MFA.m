
clc; clear all; close all;
randn('state',0); rand('state',0);

train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

OPTION = 1;  % The manifold provides a larger numbr of clusters in this option !

OPTION = 2;  % In this option I reduce the number of each digit to a max.
             % of 1000 samples -> I obtain better results (better means less clusters)

switch OPTION
    case 1
        X = train_images;
    case 2
        ndx = []; m = 1000;
        for i = 0:9
            tmp = find(train_labels==i);
            ndx = [ndx; tmp(1:m)];
        end
        X = reshape(train_images(:,ndx),[28*28,length(ndx)]);
        label = train_labels(ndx);
        %[p,n] = size(X);
    otherwise
        disp('Nothing to do')
end


k = 50*ones(50,1);
para.k = k; para.cet = 1;
para.maxit = 2500; para.num = 500;
spl = MFA_DP(X,para);

%save('Digit_Result.mat', 'spl');    % -- results obtaines with OPTION = 1;  
%save('My_Digit_Result.mat', 'spl'); % -- results obtaines with OPTION = 2;  

