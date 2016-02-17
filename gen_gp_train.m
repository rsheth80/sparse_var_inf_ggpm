function [params,f,nfe,excond,outtrace] = gen_gp_train(...
    model, data, params, meth, opts_gen)
% function [params_out, f_obj_final, num_obj_func_evals, excond, outtrace] = 
%   gen_gp_train(model, data, params, meth, opts_gen)
%
% implements coordinate ascent with minFunc except in the cases when fp_opt or 
% full_var_opt_dm_fpV are specified as optimization methods
%
% stopping conditions are specified in (optional) opts_gen structure:
%   MaxIter:    maximum no. of iterations           (default: 100)
%   optTol:     1st-order optimality condition      (default: 1e-5)
%   progTol:    absolute change in obj. func. value (default: 1e-9)
%
% model [structure]
%   mean_func
%   cov_func
%
% data [structure]
%   xt [Ntrain,D]
%   yt [Ntrain,1]
%
% params [structure]
%   hyp [structure]
%       mean
%       cov
%   var [structure]
%       xpseudo [Npseudo,D]
%       m [Npseudo,1]
%       C [Npseudo,Npseudo]
%
% meth [L x 2 cell] 
%   meth(i,:) = { objective function [handle], 
%                   optimization function options [optional structure] }
%       optimization function must take the following arguments in this order
%           optimization function [handle]
%           optimization variable [vector]
%           optimization function options [structure]
%           model 
%           data
%           params
%       optimization function options depend on algorithm used
%
% excond:
%   1, converged
%   2, exceeded max iterations
%   3, could not take a step
%
% outtrace [structure]
%   fs [L, Niters]      : history of objective function values
%   nfes [L, Niters]    : history of no. of objective function evals
%   foocs [L, Niters]   : history of first-order optimality 
%   mps [L, Niters]     : history of making progress indicator
%   ts [L, Niters]      : history of time spent in stages

N = length(data.yt);
[M,D] = size(params.var.xpseudo);
L = size(meth,1);
eyeM = eye(M);

if( nargin<5 || ~isfield(opts_gen,'optTol') )
    opttol = 1e-5;
else
    opttol = opts_gen.optTol;
end;
if( nargin<5 || ~isfield(opts_gen,'optProg') )
    progtol = 1e-9;
else
    progtol = opts_gen.optProg;
end;
if( nargin<5 || ~isfield(opts_gen,'MaxIter') )
    imax = 100;
else
    imax = opts_gen.MaxIter;
end;

iter = 0;
nfes = zeros(L,0);
fs = zeros(L,0);
foocs = zeros(L,0);
mps = zeros(L,0);
ts = zeros(L,0);
first_order_opt = inf;
making_progress = 1;
outer_mp = 0;
old_f = inf;

% does not check if input params are already optimal prior to optimization...

% compute initial obj. func. value
objfun = meth{1,1};
if(isequal(objfun,@fp_opt) || isequal(objfun,@full_var_opt_dm_fpV)) % fp and full var
    f = -calc_vlb(model,data,params);
else % all other obj. funcs.
    x0 = objfun(params);
    f = objfun(x0,model,data,params);
end;

fprintf('Iteration: %6i;  f: %4.6e\r', iter, f);
while(first_order_opt > opttol && iter < imax && making_progress)

    this_f = zeros(L,1);
    this_nfe = zeros(L,1);
    this_fooc = zeros(L,1);
    this_inner_mp = zeros(L,1);
    this_t = zeros(L,1);
    for i = 1:L

        this_t(i) = cputime;

        objfun = meth{i,1};
        opts = meth{i,2};

        if(isequal(objfun,@fp_opt)) % run fp_opt instead of minFunc
            
            [params,output,f] = fp_opt(model,data,params,opts);
            f = -f;

        elseif(isequal(objfun,@full_var_opt_dm_fpV)) % treat (m,V) as one variable
            [params,output,f] = full_var_opt_dm_fpV(model,data,params,opts);

        else % all other optimizations

            x0 = objfun(params);
            [x,f,~,output] = minFunc(objfun,x0,opts,...
                model,data,params);
            output.iterations = output.iterations - 1; % minFunc starts at iter = 1
            params = objfun(x,model,data,params,1);

        end;

        this_f(i) = f;
        this_nfe(i) = output.funcCount;
        this_fooc(i) = output.firstorderopt;
        this_t(i) = cputime - this_t(i);
        this_inner_mp(i) = (output.iterations>0);
    end;

    nfes = [nfes,this_nfe];
    fs = [fs,this_f];
    foocs = [foocs,this_fooc];
    mps = [mps,this_inner_mp];
    ts = [ts,this_t];
    iter = iter + 1;

    outer_mp = (abs(old_f - this_f(end)) > progtol);
    old_f = this_f(end);

    making_progress = max(this_inner_mp) & outer_mp;
    first_order_opt = max(this_fooc);

    fprintf('Iteration: %6i;  f: %4.6e (%4.6e 1st-order opt, time: %6i)\r', ...
        iter, this_f(end), first_order_opt, uint32(sum(sum(ts))) );
end;
fprintf('Iteration: %6i;  f: %4.6e (%4.6e 1st-order opt, time: %6i)\n', ...
    iter, this_f(end), first_order_opt, uint32(sum(sum(ts))) );

f = fs(end,end);
nfe = sum(sum(nfes));
if(first_order_opt <= opttol)
    excond = 1; % converged
elseif(iter>=imax)
    excond = 2; % exceeded max iterations
else
    excond = 3; % could not take a step
end;

outtrace.nfes = nfes;
outtrace.fs = fs;
outtrace.foocs = foocs;
outtrace.mps = mps;
outtrace.ts = ts;

% calculate posterior
params.post = calc_post_params(model,params);
