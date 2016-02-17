function [f,df] = f_obj_primal_C(x,model,data,params_in,dummy_arg)
% function [f,df] = f_obj_primal_C(x,model,data,params)
% function x0 = f_obj_primal_C(params)
% function params = f_obj_primal_C(x,model,data,params,dummy_arg)

if(nargin == 5)
    M = model.Ninducing;
    mask = zeros(M);
    mask(find(triu(reshape(1:M^2,M,M)))) = 1;
    params = params_in;
    params.var.C = reshape(x,M,M);
    params.var.C = mask.*params.var.C;
    f = params;
    df = [];
    return;
end;

if(nargin == 1)
    f = x.var.C(:);
    df = [];
    return;
end;

M = model.Ninducing;
mask = zeros(M);
mask(find(triu(reshape(1:M^2,M,M)))) = 1;
params = params_in;
params.var.C = reshape(x,M,M);
params.var.C = mask.*params.var.C;
[vlb,calcs] = calc_vlb(model,data,params);
f = -vlb;
dC = dvlb_dC(model,data,params,calcs);
df = -dC(:);
