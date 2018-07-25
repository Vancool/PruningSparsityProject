%the SFS algm 
function [validIndexMatrix,vNum,vE]=SFS_Fun(X,Y,vX,vY)
	[oX,oW]=schmidtFun(X,Y,0);%regular
	P=sortedInput(oW);
	[validIndexMatrix,vE,vNum]=MinimizeVError(oW,vX,vY,P(2,:));
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
function [validIndexMatrix,vE,vInputNum]=MinimizeVError(W,vX,vY,xIndexArray)
	n=length(xIndexArray);% the input size
	[m,validationSize]=size(vX);
    vX=[vX;ones(1,validationSize)];
    m=m+1;
    [vX,~]=schmidtFun(vX,vY,1);
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