function [f,df,ddf] = f_obj_primal_m(x,model,data,params_in,dummy_arg)
% function [f,df,ddf] = f_obj_primal_m(x,model,data,params)
% function x0 = f_obj_primal_m(params)
% function params = f_obj_primal_m(x,model,data,params,dummy_arg)
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
    f = params_in;
    f.var.m = x;
    df = [];
    ddf = [];
    return;
end;

if(nargin == 1)
    f = x.var.m;
    df = [];
    ddf = [];
    return;
end;

params = params_in;
params.var.m = x;
[vlb,calcs] = calc_vlb(model,data,params);
f = -vlb;
[df,ddf] = dvlb_dm(model,data,params,calcs);
df = -df;
ddf = -ddf;
