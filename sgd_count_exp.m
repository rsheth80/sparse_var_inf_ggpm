function sgd_count_exp(L, M, K)
% function sgd_count_exp(L, M, K)
%
% runs stochastic gradient descent (sampling w/ replacement) on blogfeedback 
% data set
%
% L: mini-batch size
% M: inducing set size
% K: number of trials [1]
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


if(nargin<3)
    K = 1;
end;

init_rand_seed(10);

fn = './data_all.mat';
load(fn); % assume training data has already been shuffled
max_count = 50;
ytrain(find(ytrain>max_count))=max_count;
ytest(find(ytest>max_count))=max_count;
Ntrain = length(ytrain);

% normalize
[~,zmu,zsg] = zscore(xtrain);
ixx = find(zsg);
oo = ones(size(xtrain,1),1);
xtrain(:,ixx) = (xtrain(:,ixx)-oo*zmu(ixx))./(oo*zsg(ixx));
oo = ones(size(xtest,1),1);
xtest(:,ixx) = (xtest(:,ixx)-oo*zmu(ixx))./(oo*zsg(ixx));

start_hyp_opt = inf;    % no hyp opt
sf_mC=2e-3;             % empirically determined
sf_h=1e-6;

% set up model and initialize GP parameters
model.mean_func = {@meanConst};
model.cov_func = {@covSEiso};
hyp.lik = [];
model.type = 'sparse_gppr';
Ninducing = M;
ix = random_sample(Ninducing,Ntrain);
d.xt = xtrain;
d.yt = ytrain;
hyp.cov=[zeros(2,1)];
hyp.mean=0;
[model,~,params] = define_init_model(model,d,hyp,Ninducing,ix);
params.hyp = minimize(params.hyp, @gp, -100, @infLaplace, model.mean_func, model.cov_func, model.lik_func, xtrain(ix,:), ytrain(ix));
params.post = calc_post_params(model,params);
paramsi = params;

mask = zeros(M);
mask(find(triu(reshape(1:M^2,M,M)))) = 1;

% run batch on full data set for comparison
NN=Ntrain;
full_data.xt = xtrain(1:NN,:);
full_data.yt = ytrain(1:NN);
define_minfunc_options;
meths = {@f_obj_primal_m,opts_m;@fp_opt,opts_V;};
tic
[params_full,efull3,nfe,excond,outtrace] = gen_gp_train(model,full_data,params,meths);
fulltimewall=toc;
fulltimecpu=sum(sum(outtrace.ts));
[efull,efull2]=calc_error_metric(model,params_full,xtest,ytest);

ix = random_sample(L,Ntrain);
iter = 1;
aa = 0;
bb = 1;
nbatches=500;           % iteration limit
e = zeros(nbatches,K);  % mean fractional error on test
e2 = zeros(nbatches,K); % mnlpd on test
e3 = zeros(nbatches,K); % vlb on train

sgdtimecpu = zeros(nbatches,K);
sgdtimewall = zeros(nbatches,K);
for k=1:K
    iter=1;
    params=paramsi;
    [e(1,k),e2(1,k)] = calc_error_metric(model,params,xtest,ytest);
    e3(1,k) = calc_vlb(model,full_data,params);
    fprintf(1,'(%3d,%2d) : %f\n',0,k,e2(1,k));
    while(iter<=nbatches)
        sgdtime0 = cputime;
        tic

        eta = (iter+aa)^(-bb);

        [~,~,calcs] = compute_approx_margs(model,[],params);

        % global terms
        dvlbdm_kld = -calcs.c2;
        A = calcs.L'\(calcs.L\(params.var.C'));
        dvlbdC_kld = diag(1./diag(params.var.C)) - mask.*(A');
        if(iter>start_hyp_opt)
            [dvlbdh_kld,calcs4] = dvlb_dh_kld(model,[],params,calcs);
        end;

        % local terms
        dvlbdm_lik = zeros(M,length(ix));
        dvlbdC_lik = zeros(M,M,length(ix));
        if(iter>start_hyp_opt)
            dvlbdh_lik = zeros(length(dvlbdh_kld),length(ix));
        end;
        for l = 1:length(ix)
            d.xt = xtrain(ix(l),:);
            d.yt = ytrain(ix(l));
            [pred_mean,pred_var,calcs2] = compute_approx_margs(model,d,params,calcs);

            [expdf,calcs3] = calc_expf(1,model,d,params,pred_mean,pred_var);
            dvlbdm_lik(:,l) = calcs2.C18*expdf;

            expdf2 = calc_expf(2,model,d,params,pred_mean,pred_var,calcs3);
            dvlbdC_lik0 = calcs2.C18*bsxfun(@times,calcs2.C18',expdf2);
            dvlbdC_lik(:,:,l) = mask.*(params.var.C*dvlbdC_lik0);

            if(iter>start_hyp_opt)
                dvlbdh_lik(:,l) = dvlb_dh_lik(model,d,params,calcs4);
            end;
        end;

        params.var.m = params.var.m + sf_mC*eta*(dvlbdm_kld + Ntrain*mean(dvlbdm_lik,2));
        params.var.C = params.var.C + sf_mC*eta*(dvlbdC_kld + Ntrain*mean(dvlbdC_lik,3));
        params.post = calc_post_params(model,params);
        
        sgdtimewall(iter,k) = toc;
        sgdtimecpu(iter,k) = cputime-sgdtime0;

        iter = iter + 1;
        ix = random_sample(L,Ntrain);
        [e(iter,k),e2(iter,k)] = calc_error_metric(model,params,xtest,ytest);
        e3(iter,k) = calc_vlb(model,full_data,params);
        fprintf(1,'(%3d,%2d) : %f\n',iter,k,e2(iter,k));
    end;
end;

keyboard;

function [e,e2] = calc_error_metric(model,params,xtest,ytest)

error_metric = 'mfe';

datar = gen_gp_predict(model,[],params,xtest);
dtest.xs = xtest;
dtest.ys = ytest;
e = compute_error_metric(model,dtest,params,datar,error_metric);
e2 = compute_error_metric(model,dtest,params,datar,'mnlpd');
