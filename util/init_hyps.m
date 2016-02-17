function hyp = init_hyps(model,x,y)

model_ltype = strrep(strrep(model.type,'sod_',''),'sparse_','');

switch(lower(model_ltype))
case 'gppr' % assume covSEiso, meanConst, likPoisson/exp
    hyp = init_gppr_hyp_model1(x,y);
    hyp.lik = [];
case 'gpc' % assume covSEiso, meanZero, likLogistic
    hyp = init_gpc_hyp_model1(x,y);
    hyp.lik = [];
case 'gpo' % assume covSEiso, meanZero, likCumLog
    hyp = init_default(model,x,y);
    hyp.cov(1) = log(max(max(sqrt(sq_dist(x'))))/2); 
    n = eval(feval(model.lik_func{:}));
    delta = 2;
    slope = 50;
    hyp.lik = [-1;log(ones(n-2,1)*delta/(n-2));log(slope)];
    ptemp.hyp = hyp;
    phi = ord_disp_hyp(ptemp);
    hyp.cov(2) = log(std(phi(2:end-1)));
otherwise
    hyp = init_default;
end;

%%%

function hyp = init_default(model,x,y)

D = size(x,2);

% mean hyps
n = eval(feval(model.mean_func{:}));
if(n)
    hyp.mean = zeros(n,1);
else
    hyp.mean = [];
end;

% cov hyps
n = eval(feval(model.cov_func{:}));
if(n)
    hyp.cov = log(ones(n,1));
else
    hyp.cov = [];
end;

% lik hyps
n = eval(feval(model.lik_func{:}));
if(n)
    hyp.lik = log(ones(n,1));
else
    hyp.lik = [];
end;

%%%

function hyp = init_gpc_hyp_model1(x,y)

my = mean(y/2+0.5); % transform range from {-1,+1} to {0,1}
if(my<=0 || my>=1)
    mn = 0;
else
    mn = log(my/(1-my));
end;
if(isnan(mn)||isinf(mn))
    mn = 0;
end;
sf2 = 1;
el = max(max(sqrt(sq_dist(x'))))/2; 
if(~el)
    el = 1;
end;
%hyp.mean = mn;
hyp.mean=[];
%el=1;sf2=1;
hyp.cov = log([el;sqrt(sf2)]);
hyp.lik = [];

%hyp.mean=[];
%hyp.cov=log([2.85;2.35;]);

function hyp = init_gppr_hyp_model1(x,y)

my = mean(y);
vy = var(y);
if(my<=0 || vy<=my)
    sf2 = 1;
else
    sf2 = log((vy-my)/my^2 + 1);
end;
if(my==0)
    mn = 1;
else
    mn = log(my) - 0.5*sf2;
end;
if(mn<=0)
    mn = 1;
end;
el = max(max(sqrt(sq_dist(x'))))/2; 
if(~el)
    el = 1;
end;
hyp.mean = log(mn);
hyp.cov = log([el;sqrt(sf2)]);
hyp.lik = [];
