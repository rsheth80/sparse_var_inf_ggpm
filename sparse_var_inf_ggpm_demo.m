function demo(mdt, frac)
% function demo(model_data_type, subset_fraction)
%
% model_data_type in {'sparse','sod'}
% subset_fraction in [0,1]
%
% this function provides an example of how to use the sparse variational 
% inference for generalized GP models Matlab code. the example is a binary 
% classification task and is taken from the GPML toolbox website 
% (http://www.gaussianprocess.org/gpml/code/matlab/doc/)
% here we use the covSEiso kernel and Bernoulli-logit likelihood
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

if(nargin<2)
    error('Two arguments required.');
end;

rng(0);                             % set seed for RNG
figure;
axis([-4 4 -4 4]);

% ---------------------------------------------------------
% ------- original code from GPML toolbox website ---------
% ---------------------------------------------------------
n1 = 80; n2 = 40;                   % number of data points from each class
S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices
m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means

x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);

x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
plot(x1(1,:), x1(2,:), 'b+'); hold on;
plot(x2(1,:), x2(2,:), 'r+');

[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
t = [t1(:) t2(:)]; n = length(t);

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
% ---------------------------------------------------------
% ------- original code from GPML toolbox website ---------
% ---------------------------------------------------------

% ---------------------------------------------------------
% ---- sparse var inf for ggpm training and prediction ----
% ---------------------------------------------------------
model.mean_func = {meanfunc};       % set up (model, data, params)
model.cov_func = {covfunc};
hyp.lik = [];
data.xt = x;
data.yt = y;
model.type = [mdt,'_gpc'];          % train/predict with sparse or sod model
Ninducing = floor((n1+n2)*frac);    % size of subset/inducing set
[model,data,params] = define_init_model(model,data,hyp,Ninducing);
likfunc = model.lik_func;

xx = bsxfun(@plus, bsxfun(@times, data.xt(model.ix_inducing,:), data.zsg), ...
    data.zmu);
plot(xx(:,1),xx(:,2),'ko');         % identify subset/inducing set on plot

opts_m.display = 'off';
opts_m.method = 'newton';           % use Newton's method for var mean
opts_h.display = 'off';
opts_h.method = 'lbfgs';            % use quasi-Newton for hyperparameters
meth =  {   @f_obj_primal_m, opts_m; ...
            @fp_opt, [];...         % use fixed point updates for var cov
            @f_obj_primal_h,opts_h;...
        };

opts_gen.MaxIter = 100;             % these are the default values in
opts_gen.optTol = 1e-5;             % gen_gp_train, but can modify here to see
opts_gen.progTol = 1e-9;            % the effects of different stopping criteria
params_out = gen_gp_train(model,data,params,meth,opts_gen);
preds = gen_gp_predict(model,data,params_out,t,1);
lp = preds.lp;
% ---------------------------------------------------------
% ---- sparse var inf for ggpm training and prediction ----
% ---------------------------------------------------------

% ---------------------------------------------------------
% ------- original code from GPML toolbox website ---------
% ---------------------------------------------------------
plot(x1(1,:), x1(2,:), 'b+'); hold on; plot(x2(1,:), x2(2,:), 'r+')
contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);
% ---------------------------------------------------------
% ------- original code from GPML toolbox website ---------
% ---------------------------------------------------------
