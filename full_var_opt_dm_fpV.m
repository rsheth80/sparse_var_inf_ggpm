function [params,output,fp_vlb_new] = full_var_opt_dm_fpV(model,data,params,opts)
% function [params,output,fp_vlb_new] = full_var_opt_dm_fpV(model,data,params,opts)

% setup
opts_m.method = 'newton';
opts_m.display = 'off';
max_iters = 500;
min_tolProg = 1e-9;
min_tolOpt = 1e-5;
p = params;
num_iters = 0;
tolProg = inf;
p.var.V = params.var.C'*params.var.C;
tolOpt = max(max(max(abs(dvlb_dV(model,data,p)))),max(abs(dvlb_dm(model,data,p))));
trace = [];
fp_vlb_new = -calc_vlb(model,data,p);

if(tolOpt < min_tolOpt) % already optimal, so return
    output.funcCount = 0;
    output.iterations = 0;
    output.firstorderopt = tolOpt;
    output.trace = [];
    return;
end;

trace.fval = fp_vlb_new;
trace.optCond = tolOpt;

% iterations
while( num_iters < max_iters && ...
    tolProg >= min_tolProg && tolOpt >= min_tolOpt )
    fp_vlb = fp_vlb_new;

    % opt m
    objfun = @f_obj_primal_m;
    x0 = objfun(p);
    [x,f,~,output] = minFunc(objfun,x0,opts_m,model,data,p);
    p = objfun(x,model,data,p,1);

    % opt V
    [p,output,f] = fp_opt(model,data,p,opts);

    % exit checking
    tolOpt = max(max(max(abs(dvlb_dV(model,data,p)))),max(abs(dvlb_dm(model,data,p))));
    fp_vlb_new = -calc_vlb(model,data,p);
    tolProg = abs(fp_vlb_new-fp_vlb);
    num_iters = num_iters + 1;

    % save trace
    trace.fval = [trace.fval; fp_vlb_new];
    trace.optCond = [trace.optCond; tolOpt];
end;

% fp exit
params = p;
output.funcCount = num_iters;
output.iterations = num_iters;
output.firstorderopt = tolOpt;
output.trace = trace;
