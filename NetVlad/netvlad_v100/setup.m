paths= yul_localPaths();
addpath(genpath('C:\Users\scien\Documents\MATLAB\Script'));
% run ('C:\Users\scien\Documents\MATLAB\vlfeat-0.9.20\toolbox\vl_setup.m');
run( fullfile(paths.libMatConvNet, 'matlab', 'vl_setupnn.m') );

addpath( genpath(paths.libReljaMatlab) );
addpath( genpath(paths.libYaelMatlab) );

addpath('datasets/');
