function dvlbdC = dvlb_dC(model,data,params,calcs)
% function dvlbdC = dvlb_dC(model,data,params,calcs)
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

if(nargin==4)
    iscalcs = 1;
else
    iscalcs = 0;
end;

Ntrain = size(data.xt,1);
M = model.Ninducing;

mask = zeros(M);
mask(find(triu(reshape(1:M^2,M,M)))) = 1;

if(~iscalcs)
    [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params);
    L = chol(Kmm,'lower'); % should have checked for non-sing Kmm before this point
    c1 = params.var.m-fmean_pseudo;
    c2 = L'\(L\c1); % inv(Kmm)*(m-fmean_pseudo)
else
    L = calcs.L;
    c2 = calcs.c2;
end;

if(~strcmpi(model.type(1:3),'spa')) % sod
    
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

else % sparse
    
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
        pred_var = calcs.pred_var;
        pred_mean = calcs.pred_mean;
        C18 = calcs.C18;
    end;

end;

% vlb gradient due to likelihood:
if(iscalcs)
    expdf2 = calc_expf(2,model,data,params,pred_mean,pred_var,calcs);
else
    expdf2 = calc_expf(2,model,data,params,pred_mean,pred_var);
end;
dvlbdC_lik0 = C18*bsxfun(@times,C18',expdf2);
dvlbdC_lik = mask.*(params.var.C*dvlbdC_lik0);

% vlb gradient due to KL divergence:
A = L'\(L\(params.var.C'));
dvlbdC_kld = diag(1./diag(params.var.C)) - mask.*(A');

% total vlb gradient
dvlbdC = dvlbdC_lik + dvlbdC_kld;
