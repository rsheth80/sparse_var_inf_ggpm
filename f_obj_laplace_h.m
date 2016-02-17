function [f,df] = f_obj_laplace_h(x,model,data,params_in,dummy_arg)
% function [f,df] = f_obj_laplace_h(x,model,data,params)
% function x0 = f_obj_laplace_h(params)
% function params = f_obj_laplace_h(x,model,data,params,dummy_arg)

if(nargin == 1)
    f = [x.hyp.mean(:);x.hyp.cov(:);x.hyp.lik(:)];
    df = [];
    return;
end;

if(nargin == 5)
    Nmean = length(params_in.hyp.mean);
    Ncov = length(params_in.hyp.cov);
    Nlik = length(params_in.hyp.lik);
    params_update = params_in;
    params_update.hyp.mean = x(1:Nmean);
    params_update.hyp.cov = x(Nmean+(1:Ncov));
    params_update.hyp.lik = x(Nmean+Ncov+(1:Nlik));
    f = params_update;
    df = [];
    return;
end;

params = params_in;
Nmean = length(params.hyp.mean);
Ncov = length(params.hyp.cov);
Nlik = length(params.hyp.lik);
params.hyp.mean = x(1:Nmean);
params.hyp.cov = x(Nmean+(1:Ncov));
params.hyp.lik = x(Nmean+Ncov+(1:Nlik));

if(~isreal(params.hyp.cov))
    f = inf;
    df = inf;
    return;
end;

%[f,df] = gp(params.hyp,@infLaplace,model.mean_func,model.cov_func,model.lik_func,data.xt,data.yt);
[~,f,df] = infLaplace(params.hyp,model.mean_func,model.cov_func,model.lik_func,data.xt,data.yt);
xx.hyp = df;
df = f_obj_laplace_h(xx); % convert from struct to column vector
