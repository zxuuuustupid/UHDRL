data=xlsread("F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\M1_G0_LA0_RA0\Sample_1\data_motor_M1_G0_LA0_RA0_20Hz_0kN.csv");
data=data(:,6)
name='M1_G0_LA0_RA0_motor_CH6'
path='CWT-1000\motor\test\anomaly'
Untitled2(data,path,name)

%sample generation
clear all
clc

data=xlsread("F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\M0_G1_LA0_RA0\Sample_1\data_gearbox_M0_G1_LA0_RA0_20Hz_0kN.csv");
data=data(:,6)
name='M0_G1_LA0_RA0_gearbox_CH16'
path='CWT-1000\gearbox\test\anomaly'
Untitled2(data,path,name)

%sample generation
clear all
clc

data=xlsread("F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\M0_G0_LA1_RA0\Sample_1\data_leftaxlebox_M0_G0_LA1_RA0_20Hz_0kN.csv");
data=data(:,2)
name='M0_G0_LA1_RA0_leftaxlebox_CH18'
path='CWT-1000\leftaxlebox\test\anomaly'
Untitled2(data,path,name)

%sample generation
clear all
clc

data=xlsread("F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\M0_G0_LA0_RA1\Sample_1\data_rightaxlebox_M0_G0_LA0_RA1_20Hz_0kN.csv");
data=data(:,2)
name='M0_G0_LA0_RA1_rightaxlebox_CH22'
path='CWT-1000\rightaxlebox\test\anomaly'
Untitled2(data,path,name)

%sample generation
clear all
clc




% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\数据生成\数据生成\out-data\out-data\TYPE0_leftaxlebox_CH17_out_gho_1.csv");
% data=data(:,1)
% name='shotlib_leftaxlebox_gho_1'
% path='CWT-1000\leftaxlebox\train\anomaly'
% Untitled2(data,path,name)
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\数据生成\数据生成\out-data\out-data\TYPE0_leftaxlebox_CH17_out_gho_1.csv");
% data=data(:,2)
% name='shotlib_leftaxlebox_gho_1_SECOND'
% path='CWT-1000\leftaxlebox\train\anomaly'
% Untitled2(data,path,name)
% 
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\数据生成\数据生成\out-data\out-data\TYPE0_leftaxlebox_CH17_out_gho_2.csv");
% data=data(:,1)
% name='shotlib_leftaxlebox_gho_2'
% path='CWT-1000\leftaxlebox\train\anomaly'
% Untitled2(data,path,name)
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\数据生成\数据生成\out-data\out-data\TYPE0_leftaxlebox_CH17_out_gho_3.csv");
% data=data(:,1)
% name='shotlib_leftaxlebox_gho_3'
% path='CWT-1000\leftaxlebox\train\anomaly'
% Untitled2(data,path,name)


% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\Data_Pre Stage\Data_Final_Stage\Training\M0_G4+G5_LA0_RA0\Sample_1\data_gearbox.csv");
% data=data(:,6)
% name='TYPE89_gearbox_CH15'
% path='CWT-1000\gearbox\test\anomaly\anomalyTYPE89\anomaly'
% Untitled2(data,path,name)

% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\Data_Pre Stage\Data_Final_Stage\Training\M0_G0_LA1+LA2+LA3+LA4_RA0\Sample_1\data_leftaxlebox.csv");
% data=data(:,2)
% name='TYPE13141516_leftaxlebox_CH17'
% path='CWT-1000\leftaxlebox\test\anomaly\anomalyTYPE13141516\anomaly'
% Untitled2(data,path,name)
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\Data_Pre Stage\Data_Final_Stage\Training\M1_G0_LA1_RA0\Sample_1\data_gearbox.csv");
% data=data(:,6)
% name='TYPE113_gearbox_CH15'
% path='CWT-1000\gearbox\test\anomaly\anomalyTYPE113\anomaly'
% Untitled2(data,path,name)
% 
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\Data_Pre Stage\Data_Final_Stage\Training\M1_G0_LA1_RA0\Sample_1\data_rightaxlebox.csv");
% data=data(:,2)
% name='TYPE113_rightaxlebox_CH20'
% path='CWT-1000\rightaxlebox\test\anomaly\anomalyTYPE113\anomaly'
% Untitled2(data,path,name)
% 
% 
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\Data_Pre Stage\Data_Final_Stage\Training\M1_G0_LA1_RA1\Sample_1\data_gearbox.csv");
% data=data(:,6)
% name='TYPE11317_gearbox_CH15'
% path='CWT-1000\gearbox\test\anomaly\anomalyTYPE11317\anomaly'
% Untitled2(data,path,name)
% 
% 
% clear all
% clc
% 
% data=xlsread("D:\science\PHM\PHM比赛数据集\Data_Pre Stage\Data_Final_Stage\Training\M0_G3_LA1_RA0\Sample_1\data_rightaxlebox.csv");
% data=data(:,2)
% name='TYPE713_rightaxlebox_CH20'
% path='CWT-1000\rightaxlebox\test\anomaly\anomalyTYPE713\anomaly'
% Untitled2(data,path,name)




