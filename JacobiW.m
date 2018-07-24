%count jacobi matrix of output error respect to weight
function G=JacobiW(predictMatrix,groundTrueMatrix,hidden_after,X,normMatrix)
	[m,n]=size(predictMatrix);
	[a,b]=size(X);
	dataSize=n;
	temp=1/dataSize*(predictMatrix- groundTrueMatrix);
	temp=temp*(ones(n,n)./normMatrix);
	k=sigmoidDerivative(hidden_after);
	temp=temp.*k;
	G=zeros(m,a);
    3
	for i=1:n
		thisTemp=temp(:,i);
		for j=1:m
			for z=1:a
				G(j,z)=thisTemp(j)*X(z,i)+G(j,z);
			end
		end
	end
end