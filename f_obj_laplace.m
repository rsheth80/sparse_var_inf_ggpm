function [f,df,ddf] = f_obj_laplace(x,model,data,params_in,dummy_arg)
% function [f,df,ddf] = f_obj_laplace(x,model,data,params)
% function x0 = f_obj_laplace(params)
% function params = f_obj_laplace(x,model,data,params,dummy_arg)
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

if(nargin == 5)
    f = params_in;
    f.var.m = x;
    K = compute_model(model,data,f);
    [~,~,d2lp] = feval(model.lik_func{:}, ...
        params_in.hyp.lik, data.yt, f.var.m, [], 'infLaplace');
    sW = sqrt(-d2lp);
    L = chol(eye(size(K)) + sW*sW'.*K);
    A  = L'\bsxfun(@times,sW,K);
    V = K - A'*A;
    f.var.C = chol(V);
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
[lp,d1lp,d2lp] = feval(model.lik_func{:}, ...
    params.hyp.lik, data.yt, params.var.m, [], 'infLaplace');
[Kmm,~,~,fmean_pseudo] = compute_model(model,data,params);
L = chol(Kmm,'lower');
d1 = params.var.m - fmean_pseudo;
d2 = L'\(L\d1);
f = -(-0.5*d1'*d2 + sum(lp));
df = -(-d2 + d1lp);
ddf = -(-L'\(L\eye(size(L))) + diag(d2lp));
