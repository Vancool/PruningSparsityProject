clc,clear
trainMatrix=load('trainMatrix.mat');
trainMatrix=trainMatrix.trainMatrix;
trainLabelMatrix=load('trainLabelMatrix.mat');
trainLabelMatrix=trainLabelMatrix.trainLabelMatrix;
validMatrix=load('validMatrix.mat');
validMatrix=validMatrix.validMatrix;
validLabelMatrix=load('validLabelMatrix.mat');
validLabelMatrix=validLabelMatrix.validLabelMatrix;
%get train 200 validation 100 
trainMatrix=trainMatrix(:,1:500);
trainLabelMatrix=trainLabelMatrix(:,1:500);
validMatrix=validMatrix(:,1:200);
validLabelMatrix=validLabelMatrix(:,1:200);
[validIndexMatrix,W,f_E,v_E,vNum]=preTraining(trainMatrix,trainLabelMatrix,validMatrix,validLabelMatrix);