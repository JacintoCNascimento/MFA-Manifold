
clc; clear all; close all;
randn('state',0); rand('state',0);
load('Face_Result.mat');


T = length(spl.A);

mu1 = cell(T,1); A1 = cell(T,1);
for t = 1:T
    mu1{t} = spl.mu{t} + spl.A{t}*diag(spl.z{t}.*spl.w{t})*spl.S1{t};
    Lambda = spl.S2{t}-spl.S1{t}*spl.S1{t}'; 
    L = chol(Lambda+realmin*eye(size(spl.S2{t},1)));
    A1{t} = spl.A{t}*diag(spl.z{t}.*spl.w{t})*L';
end

p = size(A1{1},1); 
num = [5:10:150]; Psi = cell(length(num),1);
for j = 1:length(num)
    Psi{j} = randn(num(j),p)/sqrt(p);
end

figure(1); 
subplot(2,2,[1,2]); imagesc(spl.H); colormap cool; colorbar; title('Cluster occupation')
xlabel('Sample index'); ylabel('Cluster index');
subplot(2,2,[3,4]); bar(1:T,spl.qai); title('\lambda(t)'); 
xlabel('Cluster index'); ylabel('Probability of usage');

[vv,nn] = sort(-spl.qai); tt = 11; k = 50*ones(50,1); 
for t = 1:tt
    figure(2); subplot(3,4,t); imagesc(reshape(mu1{nn(t)},[64,64])); colormap gray; axis off; title(['Cluster ' num2str(nn(t))]); 
%     title(['Mean for cluster ' num2str(nn(t))]); % ylim([0 1])
    figure(3); subplot(3,4,t); bar(1:k(nn(t)),spl.z{nn(t)}.*spl.w{nn(t)},'k'); axis([0 k(nn(t)) -5 5]); title(['Cluster ' num2str(nn(t))]); 
%     title(['z\circ{}w for cluster ' num2str(nn(t))]); 
end

% Generate testing signals:
rd = randn('state'); rdn = rand('state');
randn('state',0); rand('state',0);
load Face_data.mat
[p0,n0] = size(images);
ord = randperm(n0);
randn('state',rd); rand('state',rdn);

Y = images(:,ord((n0-49):n0));  
labelt = [lights(ord((n0-49):n0))' poses(:,ord((n0-49):n0))']; labet = labelt(:,1);
[pt,nt] = size(Y); 

Err = zeros(length(num),1);
for j = 1:length(num)
    [Y2{j},tt1{j}] = MFA_CS(Psi{j}*Y,Psi{j},A1,mu1,spl.Phi,spl.qai);
    Err(j) = norm(Y-Y2{j},'fro')/norm(Y-0,'fro');
    disp([num2str(j) '/' num2str(length(num)) ' Projections: ' num2str(num(j)) ...
           ' Errors: ' num2str(Err(j))]);
end

figure(4);  plot(num/64/64*100,Err(:,1),'ko-'); title('Relative reconstruction Error'); 
xlabel('Percentage of measurement'); ylabel('Relative reconstruction error')

% ord1 = randperm(size(Y,2)+2); ord1 = ord1 - 2; ord1(ord1<1) = []; 
ord1 = randperm(size(Y,2));
vv = 10; 
for i = 1:25
    figure(5); subplot(5,5,i); imagesc(reshape(Y(:,ord1(i)),[64,64])); colormap gray; axis off;
    figure(6); subplot(5,5,i); imagesc(reshape(Y2{vv}(:,ord1(i)),[64,64])); colormap gray; axis off;
end
num(vv)/64/64*100

