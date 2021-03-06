function datar = gen_gp_predict(model,data,params,xs,ys)
% function predictions = gen_gp_predict(model,data,params,xs)
% function predictions = gen_gp_predict(model,data,params,xs,ys)
%
% xs is Ntest x D test locations
% predictions is a structure:
%   x:          [Ntest x D]     equal to xs
%   f_mean:     [Ntest x 1]     posterior GP mean evaluated at xs
%   f_var:      [Ntest x 1]     posterior GP variance evaluated at xs
%   y_mean:     [Ntest x 1]     output mean evaluated at xs
%   y_var:      [Ntest x 1]     output variance evaluated at xs
%   y_mode:     [Ntest x 1]     output mode evaluated at xs (only for gpo)
%
% if test outputs, ys, are supplied, then predictions will have the 
% additional field
%   lp:         [Ntest x 1]     log probability
%
%   rs  022116  corrected log prob output in lp field of predictions
%               added ys as an optional argument
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

model_ltype = strrep(strrep(model.type,'sod_',''),'sparse_','');

Kii = feval(model.cov_func{:}, params.hyp.cov, xs, 'diag');
Knm = feval(model.cov_func{:}, params.hyp.cov, xs, params.var.xpseudo);
fmean_test = feval(model.mean_func{:}, params.hyp.mean, xs);

% latent mean/var
pred_mean = fmean_test + Knm*params.post.alpha;
pred_var = Kii + sum(Knm*params.post.B.*Knm,2); % B should be neg semi def

% response mean/var
if(nargin<5)
    [~,y_pred_mean,y_pred_var]=feval(model.lik_func{:},params.hyp.lik,[], ...
                                    pred_mean,pred_var);
else
    [lp,y_pred_mean,y_pred_var]=feval(model.lik_func{:},params.hyp.lik,ys, ...
                                    pred_mean,pred_var);
end;

% log probs
if(strcmp(model_ltype(1:3),'gpo'))
    y_pred_mode = zeros(size(xs,1),1);
    Lcat = model.lik_func{2};
    lp = zeros(size(xs,1),1);
    lpi = zeros(1,Lcat);
    for i = 1:size(xs,1)
        lpi = feval(model.lik_func{:},params.hyp.lik,1:Lcat,pred_mean(i), ...
                    pred_var(i));
        [lp(i),y_pred_mode(i)] = max(lpi);
    end;
end;

datar = struct;
datar.x = xs;
datar.f_mean=pred_mean;
datar.f_var=pred_var;
datar.y_mean=y_pred_mean;
datar.y_var=y_pred_var;
if(strcmp(model_ltype(1:3),'gpo'))
    datar.y_mode = y_pred_mode;
end;
if(nargin>4)
    datar.lp = lp;
end;
