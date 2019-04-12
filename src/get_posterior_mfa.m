function [A1,mu1]=get_posterior_mfa(spl)
    T=size(spl.w{1},1);
    mu1 = cell(T,1); A1 = cell(T,1);
    for t = 1:T
        mu1{t} = spl.mu{t} + spl.A{t}*diag(spl.z{t}.*spl.w{t})*spl.S1{t};
        Lambda = spl.S2{t}-spl.S1{t}*spl.S1{t}';
        L = chol(Lambda+realmin*eye(size(spl.S2{t},1)));
        A1{t} = spl.A{t}*diag(spl.z{t}.*spl.w{t})*L';
    end
end
