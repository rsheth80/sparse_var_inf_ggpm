function [varargout] = likCumLog(K, hyp, y, mu, s2, inf, i)
% cumulative logit density for use with ordinal data
%
% written for use with GPML toolbox
%
% K is number of ordered categories (it is assumed that K>2)
% y is in {1,...,K}
% p(y|f) = sigmoid(slope*(phi_y - f)) - sigmoid(slope*(phi_{y-1} - f))
%
% hyp: [phi_1; log(\delta_2); log(\delta_3); ... log(\delta_{K-1}); log(slope)]
%
% phi_2 = phi_1 + exp(log(\delta_2))
% phi_3 = phi_2 + exp(log(\delta_3))
% ...
% phi_{K-1} = phi_{K-2} + exp(log(\delta_{K-1}))
%
% can call as feval({@likCumLog,K},...)
%
% rs    02/10/16    using better approximation for log(pr) when pr is very small
% rs    02/21/16    adjusted dimensions of output lp

if(nargin<4) % return no. of hyps
    varargout = {num2str(K)};
    return;
end;

smallest_arg = -15;

% constrain 
y(y<1) = 1;
y(y>K) = K;

% transform hyps 
phi = cumsum([hyp(1);exp(hyp(2:end-1));]);
phi = [-Inf;phi;Inf];
slope = exp(hyp(end));

if(nargin<6) % prediction
    if(length(y)==0)
        y = zeros(size(mu));
    end;
    if(isempty(s2)||all(s2==0)) % evaluate log prob at input y with f = mu
	arg_z = slope*(phi(y+1)-mu);
	arg_x = slope*(phi(y)-mu);
	ix_z = find(arg_z > smallest_arg);
	ix_x = find(arg_x > smallest_arg);
	lp_term_z = arg_z;
	lp_term_z(ix_z) = log(sigmoid(arg_z(ix_z)));
	lp_term_x = arg_x;
	lp_term_x(ix_x) = log(sigmoid(arg_x(ix_x)));
	lx = (y~=1);
        lp = zeros(size(y));
	lp(~lx) = lp_term_z(~lx);
	lp(lx) = lp_term_z(lx) + lp_term_x(lx) - arg_x(lx) ...
				+ log(1 - exp(arg_x(lx) - arg_z(lx)));
    else % evaluate pred log prob at input y
        lp = likCumLog(K, hyp, y, mu, s2, 'infEP'); % use gauss quad
    end;
    ymu = {};
    ys2 = {};
    if(nargout>1)
        % eval pred prob for y = 1...K
        n = size(mu,1);
        pp = zeros(n,K);
        for k = 1:K
            pp(:,k) = exp(likCumLog(K, hyp, k*ones(n,1), mu, s2));
        end;
        % eval E(y|xs) as sum over {1...K}
        ymu = pp*(1:K)';
        if(nargout>2)
            ys2 = pp*(((1:K)').^2) - ymu.^2;
        end;
    end;
    varargout = {lp,ymu,ys2};

else % inference
    switch(inf)
    case 'infLaplace'
        if(nargin<7) % do not need derivs wrt/ hyps
            dlp = {}; d2lp = {}; d3lp = {};
            % the following is used to calculate the gauss quad points
            % which are supplied in the mu vector by lik_epquad (as well as 
            % whatever else it is used for)
            arg_z = slope*(phi(y+1)-mu);
            arg_x = slope*(phi(y)-mu);
            ix_z = find(arg_z > smallest_arg);
            ix_x = find(arg_x > smallest_arg);
            lp_term_z = arg_z;
            lp_term_z(ix_z) = log(sigmoid(arg_z(ix_z)));
            lp_term_x = arg_x;
            lp_term_x(ix_x) = log(sigmoid(arg_x(ix_x)));
            lx = (y~=1);
            lp = zeros(size(y));
            lp(~lx) = lp_term_z(~lx);
            lp(lx) = lp_term_z(lx) + lp_term_x(lx) - arg_x(lx) ...
                        + log(1 - exp(arg_x(lx) - arg_z(lx)));
            if(nargout>1) % need d(log p)/df
                sy = sigmoid(slope*(mu-phi(y+1)));
                sy1 = sigmoid(slope*(mu-phi(y)));
                dlp = slope*(1 - sy - sy1);
                if(nargout>2) % need d2(log p)/df2
                    d2lp = (slope^2)*((sy-1).*sy + (sy1-1).*sy1);
                    if(nargout>3) % need d3(log p)/df3
                        d3lp = (slope^3)*(sy.*(1-sy).*(2*sy-1) ...
                            + sy1.*(1-sy1).*(2*sy1-1));
                    end;
                end;
            end;
            varargout = {lp,dlp,d2lp,d3lp};

        else % need derivs wrt/ hyps
            if(length(mu)==1&&length(y)~=length(mu))
                mu = mu*ones(size(y));
            end;
            if(length(y)==1&&length(y)~=length(mu))
                y = y*ones(size(mu));
            end;
            sy = sigmoid(slope*(mu-phi(y+1)));
            sy1 = sigmoid(slope*(mu-phi(y)));
            lp_dhyp = zeros(length(y),1);
            dlp_dhyp = lp_dhyp;
            d2lp_dhyp = lp_dhyp;
            if(i==1) % phi_1
                lp_dhyp = (slope^1)*(-1 + sy + sy1);
                if(nargout>1)
                    dlp_dhyp = (slope^2)*((1-sy).*sy + (1-sy1).*sy1);
                    if(nargout>2)
                        d2lp_dhyp = (slope^3)*((1-sy).*sy.*(2*(1-sy)-1) ...
                            + (1-sy1).*sy1.*(2*(1-sy1)-1)); 
                    else
                        d2lp_dhyp = {};
                    end;
                else
                    dlp_dhyp = {};
                end;
            elseif(i>1 & i<K) % t_i = log(delta_i), delta_i > 0
                lxz = y<i;
                lxe = y==i;
                lxg = y>i;
                lp_dhyp(lxz) = 0;
                lp_dhyp(lxe) = 1./(exp(slope*(phi(y(lxe)+1)-phi(y(lxe))))-1) ...
                                + sy(lxe);
                lp_dhyp(lxg) = -1 + sy(lxg) + sy1(lxg);
                lp_dhyp = lp_dhyp * (slope^1)*exp(hyp(i));
                if(nargout>1)
                    dlp_dhyp(lxz) = 0;
                    dlp_dhyp(lxe) = (1-sy(lxe)).*sy(lxe);
                    dlp_dhyp(lxg) = (1-sy(lxg)).*sy(lxg) ...
                                        + (1-sy1(lxg)).*sy1(lxg);
                    dlp_dhyp = dlp_dhyp * (slope^2)*exp(hyp(i));
                    if(nargout>2)
                        d2lp_dhyp(lxz) = 0;
                        d2lp_dhyp(lxe) = ...
                            (1-sy(lxe)).*sy(lxe).*(2*(1-sy(lxe))-1);
                        d2lp_dhyp(lxg) = ...
                            (1-sy(lxg)).*sy(lxg).*(2*(1-sy(lxg))-1) ...
                             + (1-sy1(lxg)).*sy1(lxg).*(2*(1-sy1(lxg))-1);
                        d2lp_dhyp = d2lp_dhyp * (slope^3)*exp(hyp(i));
                    else
                        d2lp_dhyp = {};
                    end;
                else
                    dlp_dhyp = {};
                end;
            else % slope = exp(log(slope)) = exp(hyp(end))
                lx1 = y==1;
                lxo = (y>1)&(y<K);
                lxk = y==K;
                asy = mu - phi(y+1);
                asy1 = mu - phi(y);
                lp_dhyp(lx1) = -sy(lx1).*asy(lx1);
                lp_dhyp(lxo) = asy1(lxo) + (phi(y(lxo)+1)-phi(y(lxo))) ...
                    ./(exp(slope*((phi(y(lxo)+1)-phi(y(lxo)))))-1) ...
                    - sy(lxo).*asy(lxo) - sy1(lxo).*asy1(lxo);
                lp_dhyp(lxk) = (1-sy1(lxk)).*asy1(lxk);
                lp_dhyp = slope*lp_dhyp;
                if(nargout>1)
                    dlp_dhyp(lx1) = -sy(lx1)-slope*sy(lx1).*(1-sy(lx1)).*asy(lx1);
                    dlp_dhyp(lxo) = (1-sy(lxo)-sy1(lxo)) ...
                        - slope*(sy(lxo).*(1-sy(lxo)).*asy(lxo) ...
                        + sy1(lxo).*(1-sy1(lxo)).*asy1(lxo));
                    dlp_dhyp(lxk) = 1-sy1(lxk) ...
                        - slope*sy1(lxk).*(1-sy1(lxk)).*asy1(lxk);
                    dlp_dhyp = dlp_dhyp*slope;
                    if(nargout>2)
                        d2lp_dhyp(lx1) = -2*slope*(1-sy(lx1)).*sy(lx1) ...
                            - (slope^2)*sy(lx1).*(1-sy(lx1)).*(1-2*sy(lx1)) ...
                                                                .*asy(lx1);
                        d2lp_dhyp(lxo) = -2*slope*((1-sy(lxo)).*sy(lxo) ...
                            + (1-sy1(lxo)).*sy1(lxo)) ...
                            - (slope^2)*((1-sy(lxo)).*sy(lxo).*(1-2*sy(lxo)) ...
                                                                .*asy(lxo) ...
                            + sy1(lxo).*(1-sy1(lxo)).*(1-2*sy1(lxo)) ...
                                                                .*asy1(lxo));
                        d2lp_dhyp(lxk) = -2*slope*(1-sy1(lxk)).*sy1(lxk) ...
                            - (slope^2)*sy1(lxk).*(1-sy1(lxk)) ...
                                                    .*(1-2*sy1(lxk)).*asy1(lxk);
                        d2lp_dhyp = slope*d2lp_dhyp;
                    else
                        d2lp_dhyp = {};
                    end;
                else
                    dlp_dhyp = {};
                end;
            end;
            varargout = {lp_dhyp,dlp_dhyp,d2lp_dhyp};
        end;

    case 'infEP'
        varargout = cell(1,nargout);
        [varargout{:}] = lik_epquad({@likCumLog,K},hyp,y,mu,s2);
    end;
end;

% logistic sigmoid
function y = sigmoid(x)

y = 1./(1+exp(-x));

