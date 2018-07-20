%pre-training function to pruning less useful inputs
function [validIndexMatrix,W_Matrix,final_E,vE]=preTraining(X,Y,vX,vY)
	%using gpu to training
	[m,trainSize]=size(X);
	[n,~]=size(Y);
	learningRate=0.01;
	tol=2000;
	X=[X;ones(1,trainSize)];
	m=m+1;
	%count correlation matrices
	R=zeros(m,m);
	for i=1:trainSize
		R=R+X(:,i)*X(:,i)';
	end
	[Q,A]=qr(R);
	A=inv(A);
	W_Matrix=randn(n,m);
	%output before---before the sigmoid activate
	[output_before,output]=countOutput(X,W_Matrix);
	number=0;
	E=countE(Y,output);
	numArr=number;
	EArr=E;
	while(number<tol ||E>0.05)
		number=number+1;
		W_Matrix=getNewW(W_Matrix,output,output_before,Y,X,learningRate);
		[output_before,output]=countOutput(X,W_Matrix);
		E=countE(Y,output);
		numArr=[numArr number];
		EArr=[EArr E];
	end
	final_E=EArr(end);
	plot(numArr,EArr);
	%after update,begin SFS algorithm
	P=sortedInput(W_Matrix);
	[validIndexMatrix,vE]=MinimizeVError(W_Matrix,vX,vY,P(2,:),A);
end

%count E
function result=countE(groundTrue,predictValue)
	[~,m]=size(groundTrue);
	K=(groundTrue- predictValue).^2;
	K=sum(K);
	K=sum(K)/m;
	result=K;
end 
%count the net value
function [result,result2]=countOutput(X,W)
	result=W*X;
	result2=sigmoid(result);
end
%update the weight
function result=getNewW(W,output,output_before,Y,X,learningRate)
	[m,trainSize]=size(X);
	[n,trainSize]=size(Y);
	deltaW=zeros(n,m);
	temp=(2/trainSize*(Y-output)).*sigmoidDerivative(output_before);
	for z=1:trainSize
		for i=1:n
			for j=1:m
				deltaW(i,j)=temp(i,z)*X(j,z)+deltaW(i,j);
			end
		end	
	end
	result=W-learningRate*deltaW;
end
function P=sortedInput(W)
	[outputSize,inputSize]=size(W);
	%begin sort the importance of input

	temp=[];
	R=[];
	for i=1:inputSize
		temp=[temp;sum(W(:,i).^2)];
	end
	[~,sortedIndex]=sort(temp);
	P=[temp;sortedIndex];
end
%using IndexArray to find the best number of input and inputMatrix
function [validIndexMatrix,vE]=MinimizeVError(W,vX,vY,xIndexArray,A)
	n=length(xIndexArray);% the input size
	[m,validationSize]=size(vX);
	vX=vX*A;
	vE=[];
	%i -- the number of x Input
	for i=1:n
		tempX=zeros(m,validationSize);
		for j=n:n-i+1
			k=find(xIndexArray==j);
			tempX(k,:)=vX(K,:);
		end
		[~,vO]=countOutput(tempX,W);
		tempE=countE(vY,vO);
		vE=[vE,tempE];
	end
	vInputNum=find(vE==min(vE));
	vE=min(vE);
	validIndexMatrix=zeros(n,n);
	for i=n:n-vInputNum+1
		vk=find(xIndexArray==i);
		validIndexMatrix(vk,vk)=1;
	end
end