%count jacobi matrix of output error respect to weight
function G=JacobiW(predictMatrix,groundTrueMatrix,hidden_after,X,normMatrix,weight_hidden_output,isAddTwo,weight_input_hidden)
	[m,n]=size(predictMatrix);
	[inputNum,dataNum]=size(X);
	[outputNum,hiddenNum]=size(weight_hidden_output);
	dataSize=n;
	% the gradient procedure
	temp=2/n*(predictMatrix-groundTrueMatrix);
	G=zeros(hiddenNum,inputNum);
	for i=1:n
		thisTemp=temp(:,i);
		thisX=X(:,i);
		thisHid=hidden_after(:,i);
		totalVal=zeros(hiddenNum,1);
		for j=1:outputNum
			for z=1:hiddenNum
				totalVal(z)=totalVal(z)+thisTemp(j)*weight_hidden_output(j,z)*sigmoidDerivative(thisHid(z));
			end
		end
		for i=1:inputNum
			for j=1:hiddenNum	
				G(j,i)=G(j,i)+totalVal(j)*thisX(i);
			end 
		end
	end
	if(isAddTwo)
		G(j,i)=G(j,i)+2*weight_input_hidden(j,i);
	end
end