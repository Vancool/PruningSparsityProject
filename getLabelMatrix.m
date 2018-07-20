clc,clear
trainLabel=load('trainLabel.mat');
trainLabel=trainLabel.trainLabel;
validLabel=load('validLabel.mat');
validLabel=validLabel.validLabel;
testLabel=load('testLabel.mat');
testLabel=testLabel.testLabel;

testLabelMatrix=OneHot(testLabel,10);
trainLabelMatrix=OneHot(trainLabel,10);
validLabelMatrix=OneHot(validLabel,10);
save testLabelMatrix testLabelMatrix
save trainLabelMatrix trainLabelMatrix
save validLabelMatrix validLabelMatrix