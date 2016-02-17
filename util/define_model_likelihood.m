function model = define_model_likelihood(model)
% function model = define_model_likelihood(model)
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

model_ltype = strrep(strrep(model.type,'sod_',''),'sparse_','');

switch(model_ltype)
case 'gppr'
    model.lik_func = {@likPoisson 'exp'};
case 'gpprlog'
    model.lik_func = {@likPoisson 'log'};
case 'gpc'
    model.lik_func = {@likLogistic};
case 'gpo'
    if(~isfield(model,'Lcat'))
        error(['For ordinal regression, the no. of categories, Lcat, must '...
                'be defined as a field in model.']);
    end;
    model.lik_func = {@likCumLog model.Lcat};
otherwise
    model.lik_func = {};
end;
