function [X2,tt,X1,H] = MFA_CS(Y,Psi,A,mu,Phi,pai)
% MFA_CS: this function implements the CS reconstruction algorithm
% described in the paper "Compressive Sensing on Manifolds Using a
% Nonparametric Mixture of Factor Analyzers: Algorithm and Performance
% Bounds" 

% See the included file "MFA_code_doc.pdf" for usage instructions.

% Jacinto don't be silly -> Think about the more direct way to compute.

[m,n] = size(Y); [m,p] = size(Psi); T = length(mu);
H = zeros(T,n); 
EX = cell(T,1);  Ri = zeros(m,m);
for t = 1:T
   P = Psi*A{t};
   tmp = zeros(size(Psi'));
   for j = 1:size(Psi,2)
       tmp(j,:) = Psi(:,j)'/Phi{t}(j);
   end
   G = Psi*tmp + Ri; Gi = inv(G);
   D = inv(P'*Gi*P+eye(size(A{t},2)));
   Yt = Y - (Psi*mu{t})*ones(1,n);
   PY = (Gi - Gi*P*D*P'*Gi)*Yt;
   EX{t} = (A{t}*P'+tmp)*PY + mu{t}*ones(1,n);
   [L1,U1] = lu(Gi); tmp1 = real(sum(log(diag(U1))));
   [L2,U2] = lu(D); tmp2 = real(sum(log(diag(U2))));
   H(t,:) = log(pai(t)+eps) - m/2*log(2*pi) + 1/2*tmp1 + 1/2*tmp2 ...
               - 1/2*sum(PY.*Yt);
end
H = exp(H + repmat(-max(H,[],1),[T,1])); 
H = H./repmat(sum(H,1),[T,1]);

% Reconstruction.
X1 = zeros(p,n); tt = zeros(1,n); 
for i = 1:n
    [vv,tt(i)] = max(H(:,i));
    X1(:,i) = EX{tt(i)}(:,i);
end

X2 = zeros(p,n);
for i = 1:n
    for t = 1:T
        X2(:,i) = X2(:,i) + H(t,i)*EX{t}(:,i);
    end
end
return