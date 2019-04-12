
clc; clear all; close all;
randn('state',0); rand('state',0);

% Here we generate shifted Gaussian dataset. There are 900
% data points, each of dim=128. We then perform reconstruction based on
% three random projections.

mintheta=0; maxtheta=1; steptheta=0.01; M=(maxtheta-mintheta)/steptheta+1;
N = 128; thetas = [mintheta:steptheta:maxtheta];
P = 3; Psi = randn(P,N); 
numThetas = length(thetas);
projVals = zeros(3,00);

sigType = 'cosine';
ord = randperm(numThetas); X = zeros(N,M); label = zeros(M,1);
for ii = 1:M
    X(:,ii) = Mike_buildSignal(thetas(ord(ii)),N,sigType);
    label(ii) = thetas(ord(ii));
    projVals(:,ii) = Psi*X(:,ii);
end

% "para" is a struct with various parameters:
%   -- para.k is a vector with length equal to the maximum number of
%      clusters. Each element para.k(i) has a value equal to the maximum
%      subspace dimensionality for cluster #i. Below, we simply set all
%      para.k(i)=50.
%   -- para.cet, para.maxit and para.num are thrshold values and number of iterations for MCMC inference

k = 50*ones(50,1); 

para.k = k; para.cet = 1;
para.maxit = 2500; para.num = 500;
spl = MFA_DP(X,para);

%save('Gaussian_Result.mat', 'spl');

