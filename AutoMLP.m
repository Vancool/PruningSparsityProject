%auto-Encoder based on MLP
%writting neural network in MATLAB first time
%writer:vanco
clc,clear
%load data
%don't know why cannot using load to get the matrix
trainLabel=load('trainLabel.mat');
trainLabel=trainLabel.trainLabel;
trainData=load('trainMatrix.mat');
trainData=trainData.trainMatrix;
testLabel=load('testLabel.mat');
testLabel=testLabel.testLabel;
testData=load('testMatrix.mat');
testData=testData.testMatrix;
validData=load('validMatrix.mat');
validData=validData.validMatrix;
validLabel=load('validLabel.mat');
validLabel=validLabel.validLabel;
[m,trainSize]=size(trainData);
[m,testSize]=size(testData);
[m,validSize]=size(validData);

%original conference
featureNum=m;
outputNum=featureNum;
hiddenLayerNum=1;
HiddenNum=200;%only one hidden layer with 200 number
N=10000;%the iteraction number
learningRate=0.01;%the learning rate
%origin weight and bias
w_input_out=randn(outputNum,featureNum);
w_in_hidden=randn(HiddenNum,featureNum);
w_hidden_out=randn(outputNum,HiddenNum);
bias1=randn(HiddenNum,1);%hidden bias
bias2=randn(outputNum,1);%output bias
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
number=0;
lossValueKeep=[];
updateStep=[];
while number<N
	number=number+1;
	%get test  data training output
	hiddenMatrix_before=w_in_hidden*trainData+bias1;
	hiddenMatrix_after=sigmoid(hiddenMatrix_before);
	outputMatrix=w_hidden_out*hiddenMatrix_after+w_input_out*trainData;
	%count loss 
	lossValue=0;
	for i=1:trainSize
		lossTemp=sum((outputMatrix(:,i)-trainData(:,i)).^2);
		lossValue=lossValue+lossTemp;
	end
	lossValue=lossValue/trainSize;
	lossValue
	lossValueKeep=[lossValueKeep lossValue];
	updateStep=[updateStep number];
	if(lossValue<=0.01)
		%the loss value is small enough
		break;
	end
	%else update the weight and bias
	%update hidden_out and input_out
	delta_weight_hidden_out=zeros(outputNum,HiddenNum);
	deltaProcessValue=2/trainSize*sum(sum(outputMatrix-trainData,1));
	for i=1:outputNum
		for j=1:HiddenNum
			delta_weight_hidden_out(i,j)=sum(deltaProcessValue*hiddenMatrix_after(j,:));
		end
	end
	delta_weight_in_out=zeros(outputNum,featureNum);
	delta_bias_in_out=zeros(outputNum,1);
	for i=1:outputNum
		delta_bias_in_out(i)=deltaProcessValue;
	end
	for i=1:outputNum
		for j=1:featureNum
			delta_weight_in_out(i,j)=sum(deltaProcessValue*trainData(j,:));
		end
	end
	%update input_hidden
	delta_weight_in_hidden=zeros(outputNum,featureNum);
	for z=i:trainSize
		for i=1:outputNum
			for j=1:featureNum
                %这个地方的i 有点问题，不能超过200
                i,j,z
				delta_weight_in_hidden(i,j)=delta_weight_in_hidden(i,j)+deltaProcessValue*sigmoidDerivative(hiddenMatrix_before(i,z))*trainData(j,z);
			end
		end
	end
	delta_bias_in_hidden=zeros(outputNum,1);
	for i=1:outputNum
		delta_bias_in_hidden(i)=sum(deltaProcessValue*sigmoidDerivative(hiddenMatrix_before(i,:)));
	end

	%update the weight and bias
	w_in_hidden=w_in_hidden+learningRate*delta_weight_in_hidden;
	w_input_out=w_input_out+learningRate*delta_weight_in_out;
	bias1=bias1+learningRate*delta_bias_in_hidden;
	bias2=bias2+learningRate*delta_bias_in_out;
end

%after update ,output the weight and bias 




