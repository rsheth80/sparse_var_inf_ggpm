function [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params,keps)
% function [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params,keps)

if(nargin<4)
    keps = 1e-7;
end;

Kmm = feval(model.cov_func{:}, params.hyp.cov, params.var.xpseudo); 

if(nargout>1)
    if(isempty(data.xt))
        M = size(params.var.xpseudo,1);
        Kii = zeros(0,1);
        Knm = zeros(0,M);
        fmean_train = zeros(0,1);
    else
        Kii = feval(model.cov_func{:}, params.hyp.cov, data.xt, 'diag');
        Knm = feval(model.cov_func{:}, params.hyp.cov, data.xt, params.var.xpseudo);
        fmean_train = feval(model.mean_func{:}, params.hyp.mean, data.xt);
    end;
    fmean_pseudo = feval(model.mean_func{:}, params.hyp.mean, params.var.xpseudo);
end;

Kmm = Kmm + keps*eye(size(Kmm));
