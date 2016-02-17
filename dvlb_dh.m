function dvlbdh = dvlb_dh(model,data,params)
% log-space covariance hyperparameters

[Ntrain,D] = size(data.xt);
Npseudo = size(params.var.xpseudo,1);
Nmean = eval(feval(model.mean_func{:}));
Ncov = eval(feval(model.cov_func{:}));
Nlik = eval(feval(model.lik_func{:}));

dvlbdh_kld = zeros(Nmean+Ncov+Nlik,1);
dvlbdh_lik = zeros(Nmean+Ncov+Nlik,1);

V = params.var.C'*params.var.C;
[Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params);
L = chol(Kmm,'lower'); % should have checked for non-sing Kmm before this point
c1 = params.var.m-fmean_pseudo;
c2 = L'\(L\c1); % inv(Kmm)*(m-fmean_pseudo)
I = sparse(eye(Npseudo));
C9 = L'\(L\V); % inv(Kuu)*V
C17 = C9 - I; % inv(Kuu)*V-I
C21 = C17 + c2*c1'; % inv(Kuu)*V-I+inv(Kuu)*(m-fmean_pseudo)*(m-fmean_pseudo)'
C28 = V - Kmm; % V-Kuu
if(~strcmpi(model.type(1:3),'spa'))
    C18 = eye(model.Ninducing);
else
    C18 = L'\(L\Knm'); % inv(Kuu)*Kun
end;
C29 = L'\(L\C28); % inv(Kuu)*(V-Kuu)

% m_{q_i} and v_{q_i}
if(~strcmpi(model.type(1:3),'spa'))
    pred_mean = params.var.m;
    pred_var = diag(V);
else
    pred_mean = fmean_train + Knm*c2; % fmean_train+Knm*inv(Kuu)*(m-fmean_pseudo)
    pred_var = Kii + sum(C18'*C28.*C18',2); % Kim*inv(Kmm)*(V-Kmm)*inv(Kmm)*Kmi
end;
[expdf,calcs] = calc_expf(1,model,data,params,pred_mean,pred_var);
expdf2 = calc_expf(2,model,data,params,pred_mean,pred_var,calcs);

% d[m_{q_i}]/d[theta_mean], d[v_{q_i}]/d[theta_mean] = 0 by def
% d[-KL(phi||p)]/d[theta_mean]
dmM_dthmn = zeros(Npseudo,1);
dmN_dthmn = zeros(Ntrain,1);
dmq_dthmn = cellfun(@(x) zeros(Ntrain,1),cell(Nmean,1),'uniformoutput',false);
for n = 1:Nmean
    dmM_dthmn = feval(model.mean_func{:}, params.hyp.mean, params.var.xpseudo, n); % dmu
    dmN_dthmn = feval(model.mean_func{:}, params.hyp.mean, data.xt, n); % dmx
    dmq_dthmn{n} = dmN_dthmn - C18'*dmM_dthmn;
    dvlbdh_kld(n) = c2'*dmM_dthmn;
end;

% d[m_{q_i}]/d[theta_cov], d[v_{q_i}]/d[theta_cov]
% (derivs wrt/ hyperparameters are returned in log(hyperparam) by GPML
% toolbox; can scale back to hyperparam by dividing by 1/hyperparam)
% d[-KL(phi||p)]/d[theta_cov]
dKmm_dthcv = zeros(Npseudo); 
dKnm_dthcv = zeros(Ntrain,Npseudo);
dKii_dthcv = zeros(Ntrain,1);
dmq_dthcv = cellfun(@(x) zeros(Ntrain,1),cell(Ncov,1),'uniformoutput',false);
dvq_dthcv = cellfun(@(x) zeros(Ntrain,1),cell(Ncov,1),'uniformoutput',false);
A = zeros(Npseudo);
for n = 1:Ncov
    dKmm_dthcv = feval(model.cov_func{:}, params.hyp.cov, params.var.xpseudo, [], n);
    dKnm_dthcv = feval(model.cov_func{:}, params.hyp.cov, data.xt, params.var.xpseudo, n);
    dKii_dthcv = feval(model.cov_func{:}, params.hyp.cov, data.xt, 'diag', n);
    A = dKnm_dthcv - C18'*dKmm_dthcv;
    dmq_dthcv{n} = A*c2;
    dvq_dthcv{n} = dKii_dthcv + 2*sum(A*C29.*C18',2) - sum(C18'*dKmm_dthcv.*C18',2);
    A = L'\(L\dKmm_dthcv); % inv(Kuu)*dKuu
    dvlbdh_kld(n+Nmean) = 1/2*sum(sum(C21.*A',2));
end;

% d[sum_i E_{q_i}(log p(y_i|f_i))]/d[theta_mean]
for n = 1:Nmean
    dvlbdh_lik(n) = expdf'*dmq_dthmn{n};
end;

% d[sum_i E_{q_i}(log p(y_i|f_i))]/d[theta_cov]
for n = 1:Ncov
    dvlbdh_lik(n+Nmean) = expdf'*dmq_dthcv{n}+0.5*expdf2'*dvq_dthcv{n};
end;

% use gauss-hermite quad for calculating d[E_{q_i}(log p(y_i|f_i,theta_lik))]/d[theta_lik]
Ngh = 20;
[ghv,ghw] = gauher(Ngh);
f = bsxfun(@plus,ghv*sqrt(pred_var)',pred_mean');
for n = 1:Nlik
    for i = 1:Ntrain
        dvlbdh_lik(n+Nmean+Ncov) = dvlbdh_lik(n+Nmean+Ncov) + ...
            ghw'*feval(model.lik_func{:},params.hyp.lik,data.yt(i),f(:,i),[],'infLaplace',n);
    end;
end;

dvlbdh = dvlbdh_lik + dvlbdh_kld;
