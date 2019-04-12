
clc; clear all; close all;
randn('state',0); rand('state',0);

N = 128; thetas = [0:0.001:1];
P = 3; Psi = randn(P,N); 
numThetas = length(thetas);
projVals = zeros(3,900);

sigType = 'gaussian';
ord = randperm(numThetas); X = zeros(N,900); label = zeros(900,1);
for ii = 1:900
    X(:,ii) = Mike_buildSignal(thetas(ord(ii)),N,sigType);
    label(ii) = thetas(ord(ii));
    projVals(:,ii) = Psi*X(:,ii);
end
k = 50*ones(50,1); 

para.k = k; para.cet = 0.01;
para.burnin = 2000; para.num = 500; para.space = 1;
% spl = MFA_DP(X,para);
% save('Mike_Result.mat', 'spl');
load('Gaussian_Result.mat');

projVals1 = Psi*spl.X_hat; [vv,u] = max(spl.H); 
T = length(k); qai = spl.qai; % qai = cumprod([1;1-spl.U(1:T-1)]).*spl.U;
figure(2222)
subplot(1,2,1); scatter3(projVals(1,:),projVals(2,:),projVals(3,:),5,label,'filled');
title('Original signal projected into 3-Dim space')
subplot(1,2,2); scatter3(projVals1(1,:),projVals1(2,:),projVals1(3,:),3,u,'filled');
title({'Reconstructed signal projected into 3-Dim space', '(color = cluster segmentation)'})
figure(3)
subplot(1,2,1); imagesc(spl.H); colorbar; title('Clustering result')
xlabel('Sample index'); ylabel('Cluster index')
subplot(1,2,2); bar(1:T,qai); title('Probability of using this cluster')
xlabel('Cluster index'); ylabel('\pi')

return

% Modified mean and subspace:
mu1 = cell(T,1); A1 = cell(T,1);
for t = 1:T
    mu1{t} = spl.mu{t} + spl.A{t}*diag(spl.z{t}.*spl.w{t})*spl.S1{t};
    Lambda = spl.S2{t} - spl.S1{t}*spl.S1{t}'; 
    L = chol(Lambda+realmin*eye(size(spl.S2{t},1)));
    A1{t} = spl.A{t}*diag(spl.z{t}.*spl.w{t})*L';
end

p = size(A1{1},1); 
num = [1:1:20]; Psi = cell(length(num),1);

for j = 1:length(num)
    Psi{j} = randn(num(j),p)/sqrt(p);
end

figure(1); 
subplot(2,2,[1,2]); imagesc(spl.H); colormap cool; colorbar; title('Cluster occupation')
xlabel('Sample index'); ylabel('Cluster index');
subplot(2,2,[3,4]); bar(1:T,spl.qai); title('\lambda(t)'); 
xlabel('Cluster index'); ylabel('Probability of usage');

[vv,nn] = sort(-spl.qai); tt = 8; k = 50*ones(50,1); 
for t = 1:tt
    figure(2); subplot(3,3,t); plot(mu1{nn(t)},'k.'); axis([0 128 0 1]); title(['Cluster ' num2str(nn(t))]); 
%     title(['Mean for cluster ' num2str(nn(t))]);
    figure(3); subplot(3,3,t); bar(1:k(nn(t)),spl.z{nn(t)}.*spl.w{nn(t)},'k'); axis([0 k(nn(t)) -0.3 0.3]); title(['Cluster ' num2str(nn(t))]);
%     title(['z\circ{}w for cluster ' num2str(nn(t))]);
end

% Generate testing signals:
for ii = 1:100
    Y(:,ii) = Mike_buildSignal(thetas(ord(900+ii)),N,sigType);
    labelt(ii) = thetas(ord(900+ii));
end
[pt,nt] = size(Y); 

Err = zeros(length(num),1);
for j = 1:length(num)
    [Y2{j},tt1{j}] = MFA_CS(Psi{j}*Y,Psi{j},A1,mu1,spl.Phi,spl.qai);
    Err(j) = norm(Y-Y2{j},'fro')/norm(Y-0,'fro');
    disp([num2str(j) '/' num2str(length(num)) ' Projections: ' num2str(num(j))...
           ' Errors: ' num2str(Err(j))]);
end

figure(4);  plot(num/128*100,Err(:,1),'ko-'); title('Relative reconstruction Error'); 
xlabel('Percentage of measurement'); ylabel('Relative reconstruction error')

ord1 = randperm(size(Y,2)); vv = 5;
for i = 1:25;
    figure(5); subplot(5,5,i); plot(Y(:,ord1(i)),'k.');  axis([0 128 0 1])
    figure(6); subplot(5,5,i); plot(Y2{vv}(:,ord1(i)),'k.'); axis([0 128 0 1])
end
num(vv)/128*100