%Prepare data CIFAR-10
clc,clear
batch1=load('cifar10/data_batch_1.mat');
label1=batch1.labels;
dataMatrix1=zeros(32*32,10000);
batch1.data=double(batch1.data);
for j=1:10000
	for i=1:32*32
		%to grey picture
		dataMatrix1(i,j)=batch1.data(j,i)*0.299+0.7152*batch1.data(j,i+32*32)+0.0722+batch1.data(j,i+2*32*32);	
	end
end
%z-score
dataMatrix1=zscore(dataMatrix1);
%spilt the training data and testing data
%10 fold cross validation
[~,dataSize]=size(dataMatrix1);
testLabel=[];
AlltrainLabel=[];
testMatrix=[];
AlltrainMatrix=[];
testSize=dataSize*0.1
testLabel=label1(1:testSize,:);
testMatrix=dataMatrix1(:,1:testSize);
AlltrainLabel=label1(testSize+1:end,:);
AlltrainMatrix=dataMatrix1(:,testSize+1:end);
%validation and training data
[~,trainingSize]=size(AlltrainMatrix);
validLabel=[];
trainLabel=[];
validMatrix=[];
trainMatrix=[];
validSize=int32(trainingSize*0.3);
validLabel=AlltrainLabel(1:validSize,:);
validMatrix=AlltrainMatrix(:,1:validSize);
trainLabel=AlltrainLabel(validSize+1:trainingSize,:);
trainMatrix=AlltrainMatrix(:,validSize+1:trainingSize);
save validLabel validLabel
save validMatrix validMatrix
save trainLabel trainLabel
save trainMatrix trainMatrix
save testMatrix testMatrix
save testLabel testLabel
