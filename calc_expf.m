function [expf,calcs] = calc_expf(order,model,data,params,...
    pred_mean,pred_var,calcs)
% function expf = calc_expf(r,model,data,params,mq,vq)
%
% calculates r-th order expectations wrt/ Gaussian(f|mq,vq)
%
% mq, vq, expf are Nx1 where N is size of training data
% r in {0,1,2}
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

if(nargin==7)
    iscalcs = 1;
else
    iscalcs = 0;
end;

Ntrain = length(pred_mean);
model_ltype = strrep(strrep(model.type,'sod_',''),'sparse_','');

% don't expect saved calcs when computing expec of 0-th deriv
if(order==0||~iscalcs) 
    switch(lower(model_ltype))
    case 'gpc'
        Ngh = 20;
        [ghv,ghw] = gauher(Ngh);
        ghsig = sigmoid(bsxfun(@plus,ghv*sqrt(pred_var)',pred_mean'));
    case 'gpprlog'
        Ngh = 20;
        [ghv,ghw] = gauher(Ngh);
        ghsig = sigmoid(bsxfun(@plus,ghv*sqrt(pred_var)',pred_mean'));
        c = exp(0);
        ghlog = log(c+exp(bsxfun(@plus,ghv*sqrt(pred_var)',pred_mean')));
    case 'gpo'
        phi = cumsum([params.hyp.lik(1);exp(params.hyp.lik(2:end-1));]);
        phi = [-Inf;phi;Inf];
        phi_y = phi(data.yt+1);
        phi_ym1 = phi(data.yt);
        slope = exp(params.hyp.lik(end));
        Ngh = 20;
        [ghv,ghw] = gauher(Ngh);
        ghsig_y = sigmoid(slope*bsxfun(@plus,ghv*sqrt(pred_var)',...
            pred_mean'-phi_y'));
        ghsig_ym1 = sigmoid(slope*bsxfun(@plus,ghv*sqrt(pred_var)',...
            pred_mean'-phi_ym1'));
    end;
elseif(iscalcs)
    switch(lower(model_ltype))
    case 'gpc'
        ghsig = calcs.ghsig;
        ghw = calcs.ghw;
    case 'gpprlog'
        ghsig = calcs.ghsig;
        ghlog = calcs.ghlog;
        ghw = calcs.ghw;
    case 'gpo'
        ghsig_y = calcs.ghsig_y;
        ghsig_ym1 = calcs.ghsig_ym1;
        ghw = calcs.ghw;
        slope = exp(params.hyp.lik(end));
    end;
end;

% expectations of order-th derivs of log lik wrt/ approximate marginals
switch(order)
case 0
    switch(lower(model_ltype))
    case 'gppr'
        vlb_lik = data.yt'*pred_mean - sum(exp(pred_mean+0.5*pred_var)) ...
            - sum(gammaln(data.yt+1));
    case 'gpprlog'
        vlb_lik = sum((bsxfun(@times,log(ghlog),data.yt')-ghlog)'*ghw) ...
            - sum(gammaln(data.yt+1));
    case 'gpc'
        X = bsxfun(@times,ghsig-1/2,data.yt')+1/2;
        X(~X) = realmin; % don't want log(0) showing up in next line
        vlb_lik = sum(log(X)'*ghw);
    case 'gpo'
        X = ghsig_ym1 - ghsig_y;
        X(~X) = realmin; % don't want log(0) showing up in next line
        vlb_lik = sum(log(X)'*ghw);
    end;
    expf = vlb_lik;
case 1
    switch(lower(model_ltype))
    case 'gppr'
        expdf = -exp(pred_mean+0.5*pred_var) + data.yt;
    case 'gpprlog'
        expdf = (bsxfun(@times,ghsig./ghlog,data.yt')-ghsig)'*ghw;
    case 'gpc'
        expdf = bsxfun(@times,1-(bsxfun(@times,ghsig-1/2,data.yt')+1/2),...
            data.yt')'*ghw;
    case 'gpo'
        expdf = slope*(ones(size(ghsig_y)) - ghsig_y - ghsig_ym1)'*ghw;
    end;
    expf = expdf;
case 2 
    switch(lower(model_ltype))
    case 'gppr'
        expdf2 = -exp(pred_mean+0.5*pred_var);
    case 'gpprlog'
        expdf2 = (bsxfun(@times,ghsig,data.yt')./ghlog.^2 ...
            .*(ghlog.*(1-ghsig)-ghsig)-ghsig.*(1-ghsig))'*ghw;
    case 'gpc'
        expdf2 = (ghsig.*(ghsig-1))'*ghw;
    case 'gpo'
        gpo_ones = ones(size(ghsig_y));
        expdf2 = -(slope^2)*((ghsig_y.*(gpo_ones-ghsig_y) ...
            + (ghsig_ym1.*(gpo_ones-ghsig_ym1)))'*ghw);
    end;
    expf = expdf2;
end;

if(nargout==2)
    switch(lower(model_ltype))
    case 'gppr'
        if(~iscalcs)
            calcs = [];
        end;
    case 'gpprlog'
        calcs.ghsig = ghsig;
        calcs.ghlog = ghlog;
        calcs.ghw = ghw;
    case 'gpc'
        calcs.ghsig = ghsig;
        calcs.ghw = ghw;
    case 'gpo'
        calcs.ghsig_y = ghsig_y;
        calcs.ghsig_ym1 = ghsig_ym1;
        calcs.ghw = ghw;
    end;
end;

