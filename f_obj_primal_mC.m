function [f,df] = f_obj_primal_mC(x,model,data,params_in,dummy_arg)
% function [f,df] = f_obj_primal_mC(x,model,data,params)
% function x0 = f_obj_primal_mC(params)
% function params = f_obj_primal_mC(x,model,data,params,dummy_arg)
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

if(nargin == 5)
    M = model.Ninducing;
    params = params_in;
    params.var.m = x(1:M);
    mask = zeros(M);
    mask(find(triu(reshape(1:M^2,M,M)))) = 1;
    c = x(M+1:end);
    params.var.C = reshape(c,M,M);
    params.var.C = mask.*params.var.C;
    f = params;
    df = [];
    return;
end;

if(nargin == 1)
    f = [x.var.m;x.var.C(:)];
    df = [];
    return;
end;

M = model.Ninducing;
mask = zeros(M);
mask(find(triu(reshape(1:M^2,M,M)))) = 1;
params = params_in;
params.var.m = x(1:M);
c = x(M+1:end);
params.var.C = reshape(c,M,M);
params.var.C = mask.*params.var.C;
[vlb,calcs] = calc_vlb(model,data,params);
f = -vlb;
dm = -dvlb_dm(model,data,params,calcs);
dC = -dvlb_dC(model,data,params,calcs);
df = [dm;dC(:);];
