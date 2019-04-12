function Xh=draw_mfa(spl,n)

[A1,mu1]=get_posterior_mfa(spl);
z=spl.z;
K=size(z{1},1);
T=size(A1{1},2);
wh=randn(K,n);
lambda=spl.qai;

for i=1:n
    t(i) = find(mnrnd(1,lambda));
    Xh(:,i)=A1{t(i)}*(z{t(i)}.*wh(:,i))+mu1{t(i)};
end
