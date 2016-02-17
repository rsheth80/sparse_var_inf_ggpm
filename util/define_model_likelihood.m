function model = define_model_likelihood(model)

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
