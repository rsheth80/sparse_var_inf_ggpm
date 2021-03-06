function [params,output,fp_vlb_new] = fp_opt(model,data,params,opts)
% function [params,output,f] = fp_opt(model,data,params,opts)
%
% Performs fixed point updates on the matrix stored in params.var.V
%
% Note: The matrix returned in params.var.C is not the Cholesky decomposition of
% params.var.V in the sense that params.var.C is lower triangular and 
% params.var.V = params.var.C' * params.var.C (whereas params.var.C would be the
% Cholesky decomposition if params.var.V = params.var.C' * params.var.C and
% params.var.C was upper right, or params.var.V = params.var.C * params.var.C' 
% and params.var.C was lower left). This is OK since most of the code only 
% assumes params.var.C' * params.var.C = params.var.V, not that params.var.C is
% the Cholesky decomposition. The only place where the Cholesky decomposition 
% is assumed is in functions related to optimization in the Cholesky 
% decomposition (dvlb_dC.m, f_obj_primal_C.m, f_obj_primal_mC.m). Therefore, 
% either fp_opt.m and these functions should not be run together, or 
% params.var.C should be corrected to hold the actual Cholesky decomposition
% before these functions are run.
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

% fp method params (defaults matched to minfunc)
if( nargin<4 || ~isfield(opts,'optTol') )
    min_tolOpt = 1e-5;
else
    min_tolOpt = opts.optTol;
end;
if( nargin<4 || ~isfield(opts,'optProg') )
    min_tolProg = 1e-9;
else
    min_tolProg = opts.optProg;
end;
if( nargin<4 || ~isfield(opts,'MaxIter') )
    max_fp_iters = 500;
else
    max_fp_iters = opts.MaxIter;
end;

% fp initial value
V = params.var.C'*params.var.C;

% fp iteration setup
p = params;
p.var.V = V;
num_fp_iters = 0;
tolProg = inf;
tolOpt = max(max(abs(dvlb_dV(model,data,p))));
trace = [];
fp_vlb_new = calc_vlb(model,data,p);

if(tolOpt < min_tolOpt) % already optimal, so return
    output.funcCount = 0;
    output.iterations = 0;
    output.firstorderopt = tolOpt;
    output.trace = [];
    return;
end;

N = length(data.yt);
M = model.Ninducing;
eyeM = eye(M);
expdf2 = zeros(N,1);
B = zeros(M);
ciV = zeros(M);
iciV = zeros(M);

% fp calculations setup
[Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,p);

cKmm = chol(Kmm,'lower');
iKmm = cKmm'\(cKmm\eyeM);
pred_mean = fmean_train + Knm*(cKmm'\(cKmm\(params.var.m - fmean_pseudo)));

if(~strcmp(lower(model.type(1:2)),'sp')) % sod
    Kmn_tilde = eye(N);
else % sparse
    Kmn_tilde = cKmm'\(cKmm\Knm');
end;

b_N = Kii - sum(Knm.*Kmn_tilde',2);
pred_var = zeros(N,1);

trace.fval = fp_vlb_new;
trace.optCond = tolOpt;

% fp iterations
while( num_fp_iters < max_fp_iters && ...
    tolProg >= min_tolProg && tolOpt >= min_tolOpt )
    fp_vlb = fp_vlb_new;

    % fp update
    pred_var = b_N + sum(Kmn_tilde'*V.*Kmn_tilde',2);
    expdf2 = calc_expf(2,model,data,params,pred_mean,pred_var);
    B = Kmn_tilde*bsxfun(@times,Kmn_tilde',-expdf2);
    ciV = chol(iKmm+B);
    iciV = ciV\eyeM;
    V = iciV*iciV';

    % fp iterations exit checking
    p.var.V = V;
    p.var.C = iciV';

    [fp_vlb_new,calcs] = calc_vlb(model,data,p);
    tolOpt = max(max(abs(dvlb_dV(model,data,p,calcs))));
    tolProg = abs(fp_vlb_new-fp_vlb);
    num_fp_iters = num_fp_iters + 1;

    % save trace
    trace.fval = [trace.fval; fp_vlb_new];
    trace.optCond = [trace.optCond; tolOpt];
end;

% fp exit
params = p;
output.funcCount = num_fp_iters;
output.iterations = num_fp_iters;
output.firstorderopt = tolOpt;
output.trace = trace;
