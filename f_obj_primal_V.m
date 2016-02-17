function [f,df] = f_obj_primal_V(x,model,data,params_in,dummy_arg)
% function [f,df,ddf] = f_obj_primal_V(x,model,data,params)
% function x0 = f_obj_primal_V(params)
% function params = f_obj_primal_V(x,model,data,params,dummy_arg)
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
    params.var.V = reshape(x,M,M);
    f = params;
    df = [];
    return;
end;

if(nargin == 1)
    if(isfield(x.var,'V'))
        f = x.var.V(:);
    else
        V = x.var.C'*x.var.C;
        f = V(:);
    end;
    df = [];
    return;
end;

M = model.Ninducing;
params = params_in;
params.var.V = reshape(x,M,M);
[vlb,calcs] = calc_vlb(model,data,params);
f = -vlb;
dV = dvlb_dV(model,data,params,calcs);
df = -dV(:);

% if using derivativeCheck to verify, remember to remove 'C' from params.var 
% (calc_vlb won't use params.var.V if params.var.C is present)
