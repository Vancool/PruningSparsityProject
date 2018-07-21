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
trainMatrix=trainMatrix(:,1:200);
trainLabelMatrix=trainLabelMatrix(:,1:200);
validMatrix=validMatrix(:,1:100);
validLabelMatrix=validLabelMatrix(:,1:100);
[validIndexMatrix,W,f_E,v_E]=preTraining(trainMatrix,trainLabelMatrix,validMatrix,validLabelMatrix);