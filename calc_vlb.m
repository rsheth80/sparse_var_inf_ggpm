function [vlb,calcs] = calc_vlb(model,data,params)
% function [vlb,calcs] = calc_vlb(model,data,params)
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

[Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params);
try
    L = chol(Kmm,'lower');
catch
    vlb = -inf;
    calcs = {};
    return;
end;
c1 = params.var.m-fmean_pseudo;
c2 = L'\(L\c1); % inv(Kmm)*(m-fmean_pseudo)

if(~strcmpi(model.type(1:3),'spa')) % sod

    pred_mean = params.var.m;

    if(isfield(params.var,'V')&&~isfield(params.var,'C'))
        traceC9 = trace(L'\(L\params.var.V)); % trace(inv(Kmm)*V)
        logdetV = log(det(params.var.V));
        pred_var = diag(params.var.V);
    else
        x1 = L\params.var.C';
        traceC9 = sum(sum(x1.*x1));
        logdetV = 2*sum(log(abs(diag(params.var.C))));
        pred_var = sum(params.var.C.^2,1)';
    end;

    C18 = eye(model.Ninducing);

else % sparse

    C18 = L'\(L\Knm'); % inv(Kmm)*Kmn

    pred_mean = fmean_train + Knm*c2;

    if(isfield(params.var,'V')&&~isfield(params.var,'C'))
        traceC9 = trace(L'\(L\params.var.V)); % trace(inv(Kmm)*V)
        C28 = params.var.V-Kmm; % V-Kmm
        c20 = sum(C18'*C28.*C18,2); % Kim*inv(Kmm)*(V-Kmm)*inv(Kmm)*Kmi
        logdetV = log(det(params.var.V));
    else
        x1 = L\params.var.C';
        traceC9 = sum(sum(x1.*x1));
        x2 = params.var.C*C18;
        x3 = sum(x2.*x2,1)';
        c20 = x3 - sum(Knm.*C18',2);
        logdetV = 2*sum(log(abs(diag(params.var.C))));
    end;
    pred_var = Kii+c20; % diag(Knn)+diag(Knm*inv(Kmm)*(V-Kmm)*inv(Kmm)*Kmn)

end;

% vlb term arising from KL divergence
vlb_kld = 0.5*((logdetV - 2*sum(log(diag(L)))) - traceC9 ...
    - c1'*c2 + model.Ninducing);

% vlb term arising from likelihood
[vlb_lik,calcs] = calc_expf(0,model,data,params,pred_mean,pred_var);

% save some calculations 
calcs.L = L;
calcs.c1 = c1;
calcs.c2 = c2;
calcs.pred_mean = pred_mean;
calcs.pred_var = pred_var;
calcs.iKmm = L'\(L\eye(size(L)));
calcs.C18 = C18;

% full vlb (includes terms not dependent on mu, K, m, or V)
vlb = vlb_lik + vlb_kld;
