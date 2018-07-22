%pre-training function to pruning less useful inputs
function [validIndexMatrix,W,final_E,vE,A]=preTraining(X,Y,vX,vY)
	%using gpu to training
	[m,trainSize]=size(X);
	[n,~]=size(Y);
	learningRate=0.00025;
	tol=1000;
	X=[X;ones(1,trainSize)];
	m=m+1;
	%count correlation matrices
	R=zeros(m,m);
 %   R=gpuArray(R);
	for i=1:trainSize
		R=R+X(:,i)*X(:,i)';
	end
	[A,U]=lu(R);
    X=A*X;%正交化
	W_Matrix=randn(n,m);
	%output before---before the sigmoid activate
	[output_before,output]=countOutput(X,W_Matrix);
	number=0;
	E=countE(Y,output)
	numArr=number;
	EArr=E;
    sn=0;
    W=[];
    minNum=0;
    %开始训练
	while(number<tol &&E>0.1&&sn<5)
		number=number+1;
		W_Matrix=getNewW(W_Matrix,output,output_before,Y,X,learningRate);
		[output_before,output]=countOutput(X,W_Matrix);
		E=countE(Y,output);
        if(E==EArr(number))
        	sn=sn+1;
            sn
        else if E<EArr(number)
        	sn=0;
            W=W_Matrix;%记录损失最低矩阵
            minNum=number;
            else
            sn=0;      
            end
        end
		numArr=[numArr number];
		EArr=[EArr E];
        E
	end
	final_E=EArr(minNum+1);
    %绘图
	plot(numArr,EArr);
	title('epoch-Error graph')
	xlabel('epoch')
	ylabel('Error')
	%after update,begin SFS algorithm
    %用SFS 算法
	P=sortedInput(W);%得到input排序
	[validIndexMatrix,vE]=MinimizeVError(W,vX,vY,P(2,:),A);%得到有效的前validNum个input的array
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
    [m,n]=size(result);
    result=zscore(result);
    result2=[];
    for i=1:n
    	result2=[result2 mySoftmax(result(:,i))];
    end
	%result2=mySoftmax(result);
end
%update the weight
function result=getNewW(W,output,output_after,Y,X,learningRate)
	[m,trainSize]=size(X);
	[n,trainSize]=size(Y);
	deltaW=zeros(n,m);
	%deltaW=gpuArray(deltaW);
	temp=(2/trainSize*(Y-output));
	%temp=gpuArray(temp);
	H=zeros(n,trainSize);
	%H=gpuArray(H);
	for i=1:trainSize
		for j=1:n
			d=zeros(n,1);
		 	for z=1:n
				d(i)=softmaxDerivative(output_after(:,i),j,z);
			end
			d=sum(d);
			H(j,i)=d;
		end
    end

	temp=H.*temp;
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
	P=[temp';sortedIndex'];
end
%using IndexArray to find the best number of input and inputMatrix
function [validIndexMatrix,vE]=MinimizeVError(W,vX,vY,xIndexArray,A)
	n=length(xIndexArray);% the input size
	[m,validationSize]=size(vX);
    vX=[vX;ones(1,validationSize)];
    m=m+1;
	vX=A*vX;
	vE=[];
	%i -- the number of x Input
	for i=1:n
		tempX=zeros(m,validationSize);
		for j=n-i+1:n
			k=find(xIndexArray==j);
			tempX(k,:)=vX(k,:);
		end
		[~,vO]=countOutput(tempX,W);
		tempE=countE(vY,vO);
		vE=[vE,tempE];
	end
	vInputNum=find(vE==min(vE));
    vInputNum=vInputNum(1);
	vE=min(vE);
	validIndexMatrix=zeros(n,1);
	for i=n-vInputNum+1:n
		vk=find(xIndexArray==i);
		validIndexMatrix(vk)=1;
	end
end