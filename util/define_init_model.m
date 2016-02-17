function [model,data,params] = define_init_model(model,data,hyp,varargin)
% function [model,data,params] = define_init_model(model,data,hyp,Nsubset)
% function [model,data,params] = define_init_model(model,data,hyp,Nsubset,ix_subset)
%
% defines and initializes a model according to 'model.type'; can be one of:
%   {'sparse_gppr','sod_gppr','sparse_gpc','sod_gpc','sparse_gpo','sod_gpo'}
% 'model' should have fields 'mean_func', 'cov_func', and 'lik_func' (and 'Lcat'
%   equal to the number of ordinal categories if this is a model for ordinal 
%   regression)
% 'data' assumed to have 'xt' and 'yt' fields
% 'hyp' should have fields 'mean', 'cov', and 'lik'
% if 'ix_subset' is provided, it is assumed that it indexes the TRAINING set, 
%   'xt'
% the training data is z-score normalized and the normalization parameters are
%   saved within the data structure

keps = 1e-9;

if(nargin<3)
    hyp = [];
end;

if(nargin==4)
    M = varargin{1};
    ix = random_sample(M,length(data.yt));
elseif(nargin==5)
    M = length(varargin{2});
    ix = varargin{2};
end;

% normalize data
[data.xt,data.zmu,data.zsg] = zscore(data.xt);

% find data subsets
switch(lower(model.type(1:2)))
case 'sp'
    model.Ninducing = M;
    model.ix_inducing = ix;
    xsub = data.xt(ix,:);
    ysub = data.yt(ix);
case 'so'
    model.Ninducing = M;
    model.ix_inducing = 1:M;
    data.xt = data.xt(ix,:);
    data.yt = data.yt(ix);
    xsub = data.xt;
    ysub = data.yt;
end;

% initialize likelihood function
model = define_model_likelihood(model);

% initialize model hyperparameters
if(isempty(hyp))
    params.hyp = init_hyps(model,xsub,ysub);
else
    params.hyp = hyp;
end;

% initialize primal variational parameters
params.var.xpseudo = xsub;
% initialize variational distribution with Gaussian via Laplace
% inference
yinducing = ysub;
post = infLaplace(params.hyp, model.mean_func, model.cov_func, ...
    model.lik_func, params.var.xpseudo, yinducing);
mu = feval(model.mean_func{:}, params.hyp.mean, params.var.xpseudo);
Kmm = feval(model.cov_func{:}, params.hyp.cov, params.var.xpseudo); 
params.var.m = Kmm*post.alpha + mu;
W = diag(post.sW.^2);
V = (eye(size(Kmm))+Kmm*W)\Kmm + eye(size(Kmm))*keps;
params.var.C = chol(V);
