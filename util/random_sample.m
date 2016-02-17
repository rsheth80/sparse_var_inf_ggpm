function ix=random_sample(n,N)
% function ix=random_sample( n, N )
% 
% Returns 'n' random indices of an 'N' x 1 sample (indices are unique)
%
% n:			requested number of indices [scalar integer]
% N:			total number of indices [scalar integer]
% ix:			indices ['n' x 1 array integer]

ix = randperm(N,n)';
