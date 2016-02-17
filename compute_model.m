function [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params,keps)
% function [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params,keps)
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

if(nargin<4)
    keps = 1e-7;
end;

Kmm = feval(model.cov_func{:}, params.hyp.cov, params.var.xpseudo); 

if(nargout>1)
    if(isempty(data.xt))
        M = size(params.var.xpseudo,1);
        Kii = zeros(0,1);
        Knm = zeros(0,M);
        fmean_train = zeros(0,1);
    else
        Kii = feval(model.cov_func{:}, params.hyp.cov, data.xt, 'diag');
        Knm = feval(model.cov_func{:}, params.hyp.cov, data.xt, params.var.xpseudo);
        fmean_train = feval(model.mean_func{:}, params.hyp.mean, data.xt);
    end;
    fmean_pseudo = feval(model.mean_func{:}, params.hyp.mean, params.var.xpseudo);
end;

Kmm = Kmm + keps*eye(size(Kmm));
