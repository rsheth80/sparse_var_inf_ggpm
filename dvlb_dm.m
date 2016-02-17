function [dvlbdm,dvlbdmm] = dvlb_dm(model,data,params,calcs)

if(nargin==4)
    iscalcs = 1;
else
    iscalcs = 0;
end;

Ntrain = size(data.xt,1);

if(~iscalcs)
    [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params);
    L = chol(Kmm,'lower'); % should have checked for non-sing Kmm before this point
    c1 = params.var.m-fmean_pseudo;
    c2 = L'\(L\c1); % inv(Kmm)*(m-fmean_pseudo)
else
    L = calcs.L;
    c2 = calcs.c2;
end;

if(~strcmpi(model.type(1:3),'spa'))

    if(~iscalcs)
        pred_mean = params.var.m;

        if(isfield(params.var,'V')&&~isfield(params.var,'C'))
            pred_var = diag(params.var.V);
        else
            pred_var = sum(params.var.C.^2,1)';
        end;
    else
        pred_mean = calcs.pred_mean;
        pred_var = calcs.pred_var;
    end;

    C18 = eye(model.Ninducing);

else

    if(~iscalcs)
        C18 = L'\(L\Knm'); % inv(Kmm)*Kmn

        pred_mean = fmean_train + Knm*c2;

        if(isfield(params.var,'V')&&~isfield(params.var,'C'))
            C28 = params.var.V-Kmm; % V-Kmm
            c20 = sum(C18'*C28.*C18,2); % Kim*inv(Kmm)*(V-Kmm)*inv(Kmm)*Kmi
        else
            x1 = L\params.var.C';
            x2 = params.var.C*C18;
            x3 = sum(x2.*x2,1)';
            c20 = x3 - sum(Knm.*C18',2);
        end;
        pred_var = Kii+c20; % diag(Knn)+diag(Knm*inv(Kmm)*(V-Kmm)*inv(Kmm)*Kmn)
    else
        pred_mean = calcs.pred_mean;
        pred_var = calcs.pred_var;
        C18 = calcs.C18;
    end;

end;

% vlb gradient due to KL divergence:
dvlbdm_kld = -c2;

% vlb gradient due to likelihood:
if(iscalcs)
    expdf = calc_expf(1,model,data,params,pred_mean,pred_var,calcs);
else
    expdf = calc_expf(1,model,data,params,pred_mean,pred_var);
end;
dvlbdm_lik = C18*expdf;

% total vlb gradient
dvlbdm = dvlbdm_lik + dvlbdm_kld;

if(nargout==2)

    % vlb hessian due to KL divergence:
    if(~iscalcs)
        dvlbdmm_kld = -L'\(L\eye(size(L)));
    else
        dvlbdmm_kld = -calcs.iKmm;
    end;
    
    % vlb hessian due to likelihood:
    if(iscalcs)
        expdf2 = calc_expf(2,model,data,params,pred_mean,pred_var,calcs);
    else
        expdf2 = calc_expf(2,model,data,params,pred_mean,pred_var);
    end;
    dvlbdmm_lik = C18*bsxfun(@times,C18',expdf2);

    % total vlb hessian 
    dvlbdmm = dvlbdmm_kld + dvlbdmm_lik;
else
    dvlbdmm = [];
end;
