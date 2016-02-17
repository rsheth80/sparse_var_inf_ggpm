function [params,output,fp_vlb_new] = full_var_opt_dm_fpV(model,data,params,opts)
% function [params,output,fp_vlb_new] = full_var_opt_dm_fpV(model,data,params,opts)
%
% Copyright (C) 2016  Rishit Sheth

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
