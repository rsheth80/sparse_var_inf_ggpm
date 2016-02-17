function [f,df,ddf] = f_obj_primal_m(x,model,data,params_in,dummy_arg)
% function [f,df,ddf] = f_obj_primal_m(x,model,data,params)
% function x0 = f_obj_primal_m(params)
% function params = f_obj_primal_m(x,model,data,params,dummy_arg)

if(nargin == 5)
    f = params_in;
    f.var.m = x;
    df = [];
    ddf = [];
    return;
end;

if(nargin == 1)
    f = x.var.m;
    df = [];
    ddf = [];
    return;
end;

params = params_in;
params.var.m = x;
[vlb,calcs] = calc_vlb(model,data,params);
f = -vlb;
[df,ddf] = dvlb_dm(model,data,params,calcs);
df = -df;
ddf = -ddf;
