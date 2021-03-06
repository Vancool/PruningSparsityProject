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

trainData=trainData(:,1:100);% for Testing

[m,trainSize]=size(trainData);

[m,testSize]=size(testData);

[m,validSize]=size(validData);

groundTrueData=trainData;%auto-encoder

[k,gSize]=size(groundTrueData);

%original conference

featureNum=m;

outputNum=k;

hiddenLayerNum=1;

HiddenNum=200;%only one hidden layer with 200 number

N=200;%the iteraction number

learningRate=0.01;%the learning rate

tol=10e-6;

fullConnectedNum=featureNum+HiddenNum+1;%update together,fully connected

%origin weight and bias

w_in_hidden=randn(HiddenNum,featureNum+1);

w_fully_connected=randn(outputNum,fullConnectedNum);

trainData=[trainData;ones(1,trainSize)];

featureNum=m+1;

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

z=zeros(outputNum,1);

K=mean(trainData);

while number<N

	number=number+1;

	%get test  data training output

	hiddenMatrix_before=w_in_hidden*trainData;

	hiddenMinMatrix=min(hiddenMatrix_before);

	hiddenMaxMatrix=max(hiddenMatrix_before);

	normMatrix=hiddenMaxMatrix-hiddenMinMatrix;

	normMatrix=repmat(normMatrix,HiddenNum,1);

	hiddenMinMatrix=repmat(hiddenMinMatrix,HiddenNum,1);

	hiddenMatrix_before=(hiddenMatrix_before-hiddenMinMatrix)./normMatrix;

	hiddenMatrix_after=sigmoid(hiddenMatrix_before);

	xaMatrix=[hiddenMatrix_after;trainData];

	outputMatrix=w_fully_connected*xaMatrix;

	%count loss 

	lossValue=0;

	for i=1:trainSize

		lossTemp=sum((outputMatrix(:,i)-groundTrueData(:,i)).^2);

		lossValue=lossValue+lossTemp;

	end

	lossValue=lossValue/trainSize;

	lossValue

	lossValueKeep=[lossValueKeep lossValue];

	updateStep=[updateStep number];

	if(lossValue<=0.1)

		%the loss value is small enough

		break;

	end

	%OWO to update xa-output layer

	%get R and C

	R0=zeros(fullConnectedNum,fullConnectedNum);

	C0=zeros(outputNum,fullConnectedNum);

	for i=1:trainSize

		R0=R0+xaMatrix(:,i)*xaMatrix(:,i)';

		C0=C0+groundTrueData(:,i)*xaMatrix(:,i)';

	end

	R0=R0/trainSize;

	C0=C0/trainSize;

	%q-r of the R

	[Q,R]=qr(R0);

	R=R';

	C0=C0';

	%solve the lower triangle result using R

	tempResult=[];

	for i=1:outputNum

		temp=solveLowerTriangle(C0(:,i),R,tol);

		tempResult=[tempResult temp];

	end

	tempResult=tempResult';

	W0=tempResult*Q';

	%end OWO 

	%begin HWO find the best descend direction

	w_hidden_out=w_fully_connected(:,1:HiddenNum);

	G=JacobiW(outputMatrix,groundTrueData,hiddenMatrix_after,trainData,normMatrix,w_hidden_out);

    R1=zeros(featureNum,featureNum);

    for i=1:trainSize

    	R1=R1+trainData(:,i)*trainData(:,i)';

    end

	Ghwo=HWO(R1,G,tol);

	%begin MOLF to find the best z,z is  a vector

	%[resultZ,jacobiMatrix,hessianMatrix]=MOLF_Z(trainData,w_fully_connected,w_in_hidden,Ghwo,z);

 	z=ones(HiddenNum,1)*0.1;%这边是自己定的，要自己重新求Hessian矩阵

    resultZ=z;

 	resultZ=repmat(resultZ,1,featureNum);

 	delta_weight_in_hidden=resultZ.*Ghwo;

 	%end MOLF

	

	

 	w_in_hidden=w_in_hidden-delta_weight_in_hidden;

	w_fully_connected=W0;%update input+hidden to output

end



%after update ,output the weight and bias 



%BP -update way

	%else update the weight and bias

	%update hidden_out and input_out

	%delta_weight_hidden_out=zeros(outputNum,HiddenNum);

	%deltaProcessValue=2/trainSize*sum(sum(outputMatrix-trainData,1));

	%for i=1:outputNum

	%	for j=1:HiddenNum

	%		delta_weight_hidden_out(i,j)=sum(deltaProcessValue*hiddenMatrix_after(j,:));

	% 		end

	% 	end

	% 	delta_weight_in_out=zeros(outputNum,featureNum);

	% 	delta_bias_in_out=zeros(outputNum,1);

	% 	for i=1:outputNum

	% 		delta_bias_in_out(i)=deltaProcessValue;

	% 	end

	% 	for i=1:outputNum

	% 		for j=1:featureNum

	% 			delta_weight_in_out(i,j)=sum(deltaProcessValue*trainData(j,:));

	% 		end

	% 	end

	% 	%update input_hidden

	% 	delta_weight_in_hidden=zeros(HiddenNum,featureNum);

	% 	for z=i:trainSize

	% 		for i=1:HiddenNum

	% 			for j=1:featureNum              

	% 				delta_weight_in_hidden(i,j)=delta_weight_in_hidden(i,j)+deltaProcessValue*sigmoidDerivative(hiddenMatrix_before(i,z))*trainData(j,z);

	% 			end

	% 		end

	% 	end

	% 	delta_bias_in_hidden=zeros(HiddenNum,1);

	% 	for i=1:HiddenNum

	% 		delta_bias_in_hidden(i)=sum(deltaProcessValue*sigmoidDerivative(hiddenMatrix_before(i,:)));

	% 	end

	% 

	% 	%update the weight and bias

	% 	w_in_hidden=w_in_hidden+learningRate*delta_weight_in_hidden;

	% 	w_input_out=w_input_out+learningRate*delta_weight_in_out;

	% 	bias1=bias1+learningRate*delta_bias_in_hidden;

	% 	bias2=bias2+learningRate*delta_bias_in_out;









