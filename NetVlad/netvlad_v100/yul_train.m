setup;
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
paths= yul_localPaths();
dbTrain= get_ucf101('trainlist01.txt');
dbVal= get_ucf101('testlist01.txt');
lr= 0.0001;