setup;
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
paths= yul_localPaths();
dbTrain= dbTokyoTimeMachine('train');
dbVal= dbTokyoTimeMachine('val');
lr= 0.0001;