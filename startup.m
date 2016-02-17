basedir = fullfile('..');       % assumes base dir is one level up from current
run(fullfile(basedir,'gpml-matlab-v3.4-2013-11-11','gpml_startup'));
addpath(fullfile(basedir,'minFunc_2012'));
addpath(fullfile(basedir,'minFunc_2012','minFunc'));
addpath(fullfile(basedir,'minFunc_2012','minFunc','compiled'));
addpath(fullfile(pwd,'util'));
