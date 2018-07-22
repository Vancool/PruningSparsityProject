%MOLF -count the jacobi matrix and hessian matrix of output error function respect to learning rate
function [resultZ,jacobiMatrix,hessianMatrix]=MOLF_Z(X,w_f,w_hidden_output,G,hiddenNum,previousZ)
	Z=sym('Z',[hiddenNum,1]);
    
	temp=[];
	for i =1:hiddenNum
		temp=[temp;G(i,:)*Z(i)];
	end
	Op=(w_hidden_output+temp)*X;
	K=sigmoid(Op);
	xa=[K;X];
	ouput=w_f*xa;
    n=length(ouput);
    X=X(1:n-1,:);
    E=sum((X-ouput).^2);
	JA=jacobian(E,Z);
    jacobiMartrix=eval(subs(JA,Z,previousZ));
    HESSIAN=hessian(E,Z);
    hessianMatrix=eval(subs(HESSIAN,Z,previousZ));
    Hess=inv(HESSIAN);

    result=Hess*JA';
    resultZ=eval(subs(result,Z,previousZ));
end

%HWO -count the learning direction using jacobi matrix
function hMatrix=HWO(R0,G,tol)
	[Q,R]=qr(R0);
	R=R';
	G=G';
	[m,n]=size(G);
	y=[];
	for i=1:n
		y=[y solveLowerTriangle(G(:,i),R,tol)];
	end
	y=y';
	hMatrix=y*Q';
end
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
	for i=1:n
		thisTemp=temp(:,i);
		for j=1:m
			for z=1:a
				G(j,z)=thisTemp(j)*X(z,i)+G(j,z);
			end
		end
	end
end
