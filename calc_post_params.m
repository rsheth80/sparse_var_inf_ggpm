function post = calc_post_params(model,params)
% function post = calc_post_params(model,params)

Kmm = compute_model(model,[],params);
L = chol(Kmm,'lower');
fmean_pseudo = feval(model.mean_func{:}, params.hyp.mean, params.var.xpseudo);
post.alpha = L'\(L\(params.var.m-fmean_pseudo));
post.B = L'\(L\(((params.var.C'*params.var.C-Kmm)/L')/L));
