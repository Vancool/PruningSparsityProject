%auto-Encoder based on MLP
%writting neural network in MATLAB first time
%writer:vanco

%load data
trainLabel=load('trainLabel.mat');
trainData=load('trainMatrix.mat');
testLabel=load('testLabel.mat');
testData=load('testMatrix.mat');
validData=load('validMatrix.mat');
validLabel=load('validLabel.mat');
[m,trainSize]=size(trainData);
[m,testSize]=size(testData);
[m,validSize]=size(validData);

%original conference
featureNum=m;
outputNum=featureNum;
hiddenLayerNum=1;
HiddenNum=200;%only one hidden layer with 200 number
%origin weight and bias
w_input_out=randn(outputNum,featureNum);
w_in_hidden=randn(HiddenNum,featureNum);
w_hidden_out=randn(outputNum,HiddenNum);
bias1=randn(HiddenNum,1);
bias2=randn(outputNum,1);
%prepare one-hot code label
%trainOneHotMatrix=[];
%validOneHotMatrix=[];
%for i=1:trainSize
%	temp=zeros(10,1);
%	temp(trainLabel(i))=1;
%	trainOneHotMatrix=[trainOneHotMatrix temp];
%end
%for i=1:validSize
%	temp=zeros(10,1);
%	temp(validLabel(i))=1;
%	validOneHotMatrix=[validOneHotMatrix temp];
%end
%-------------------Auto-encoder doesn't need label!

%get test  data training output
hiddenMatrix=w_in_hidden*trainData+bias1;
hiddenMatrix=sigmoid(hiddenMatrix);
outputMatrix=w_hidden_out*hiddenMatrix+w_input_out*trainData;
%count loss 
lossValue=0;
for i=1:trainSize
	lossTemp=sum((outputMatrix(i,:)-trainData(i,:)).^2);
	lossValue=lossValue+lossTemp;
end
lossValue=lossValue/trainSize;



