
clc; clear all; close all;
randn('state',0); rand('state',0);

load('Digit_Result.mat');    %  -> digitis with less number of samples (each)
%load('My_Digit_Result.mat'); % -> original dataset ... 

T = length(spl.A);

mu1 = cell(T,1); A1 = cell(T,1);
for t = 1:T
    mu1{t} = spl.mu{t} + spl.A{t}*diag(spl.z{t}.*spl.w{t})*spl.S1{t};
    Lambda = spl.S2{t}-spl.S1{t}*spl.S1{t}'; 
    L = chol(Lambda+realmin*eye(size(spl.S2{t},1)));
    A1{t} = spl.A{t}*diag(spl.z{t}.*spl.w{t})*L';
%        mu1{t} = spl.mu{t};
%        A1{t} = spl.A{t}*diag(spl.z{t}.*spl.w{t});
end

p = size(A1{1},1); 

num = [5:10:95]; % number of measurements for the reconstruction
num = [5:3:95]; % number of measurements for the reconstruction

Psi = cell(length(num),1);
for j = 1:length(num)
    Psi{j} = randn(num(j),p)/sqrt(p);
end

figure(1); 
subplot(2,2,[1,2]); imagesc(spl.H); colormap cool; colorbar; title('Cluster occupation')
xlabel('Sample index'); ylabel('Cluster index');
subplot(2,2,[3,4]); bar(1:T,spl.qai); title('\lambda(t)'); 
xlabel('Cluster index'); ylabel('Probability of usage');

[vv,nn] = sort(-spl.qai); tt = 25; k = 50*ones(50,1); 
for t = 1:tt
    figure(2); subplot(5,5,t); imagesc(reshape(mu1{nn(t)},[28,28])); colormap gray; axis off; title(['Cluster ' num2str(nn(t))]); 
    % title(['Mean for cluster ' num2str(nn(t))]); % ylim([0 1])
    figure(3); subplot(5,5,t); bar(1:k(nn(t)),spl.z{nn(t)}.*spl.w{nn(t)},'k'); axis([0 k(nn(t)) -5 5]); title(['Cluster ' num2str(nn(t))]); 
%     title(['z\circ{}w for cluster ' num2str(nn(t))]); 
end

% --------------- Generate testing signals: -------------------------%
test_images = loadMNISTImages('train-images.idx3-ubyte');
test_labels = loadMNISTLabels('train-labels.idx1-ubyte');

ndt = []; m = 1000;
for i = 0:9
    tmp = find(test_labels==i);
    ndt = [ndt; tmp(1:m)];
end
Y = reshape(test_images(:,ndt),[28*28,length(ndt)]);
labet = test_labels(ndt);
[pt,nt] = size(Y); 

Err = zeros(length(num),1);
for j = 1:length(num)
    [Y2{j},tt1{j}] = MFA_CS(Psi{j}*Y,Psi{j},A1,mu1,spl.Phi,spl.qai);
    Err(j) = norm(Y-Y2{j},'fro')/norm(Y-0,'fro');
    disp([num2str(j) '/' num2str(length(num)) ' Projections: ' num2str(num(j)) ...
           ' Errors: ' num2str(Err(j))]);
end

figure(4);  plot(num/28/28*100,Err(:,1),'ko-'); title('Relative reconstruction Error'); 
xlabel('Percentage of measurement'); ylabel('Relative reconstruction error')

ord1 = randperm(size(Y,2)); vv = 9;
for i = 1:25;
    figure(vv+5); subplot(5,5,i); imagesc(reshape(Y(:,ord1(i)),[28,28])); colormap gray; axis off;
    figure(vv+6); subplot(5,5,i); imagesc(reshape(Y2{vv}(:,ord1(i)),[28,28])); colormap gray; axis off;
end
num(vv)/28/28*100