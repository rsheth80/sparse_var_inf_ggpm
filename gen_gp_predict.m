function datar = gen_gp_predict(model,data,params,xs)
% function predictions = gen_gp_predict(model,data,params,xs)
%
% xs is Ntest x D test locations
% predictions is a structure:
%   x:          [Ntest x D]     equal to xs
%   lp:         [Ntest x 1]     log probability
%   f_mean:     [Ntest x 1]     posterior GP mean evaluated at xs
%   f_var:      [Ntest x 1]     posterior GP variance evaluated at xs
%   y_mean:     [Ntest x 1]     output mean evaluated at xs
%   y_var:      [Ntest x 1]     output variance evaluated at xs
%   y_mode:     [Ntest x 1]     output mode evaluated at xs (only for gpo)

model_ltype = strrep(strrep(model.type,'sod_',''),'sparse_','');

Kii = feval(model.cov_func{:}, params.hyp.cov, xs, 'diag');
Knm = feval(model.cov_func{:}, params.hyp.cov, xs, params.var.xpseudo);
fmean_test = feval(model.mean_func{:}, params.hyp.mean, xs);

% latent mean/var
pred_mean = fmean_test + Knm*params.post.alpha;
pred_var = Kii + sum(Knm*params.post.B.*Knm,2); % B should be neg semi def

% response mean/var
[lp,y_pred_mean,y_pred_var]=feval(model.lik_func{:},params.hyp.lik,[],pred_mean,pred_var);

% log probs
if(strcmp(model_ltype(1:3),'gpo'))
    y_pred_mode = zeros(size(xs,1),1);
    Lcat = model.lik_func{2};
    for i = 1:size(xs,1)
        lp = feval(model.lik_func{:},params.hyp.lik,1:Lcat,pred_mean(i),pred_var(i));
        [~,y_pred_mode(i)] = max(lp);
    end;
end;

datar = struct;
datar.x = xs;
datar.lp = lp;
datar.f_mean=pred_mean;
datar.f_var=pred_var;
datar.y_mean=y_pred_mean;
datar.y_var=y_pred_var;
if(strcmp(model_ltype(1:3),'gpo'))
    datar.y_mode = y_pred_mode;
end;
