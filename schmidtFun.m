function [O,W]=schmidtFun(C,Y,hasOrder)
	O(:,1)=C(:,1)/norm(C(:,1));
	[m,n]=size(C);
	A=zeros(m,m);
	for i=2:m
		res=C(:,i);
		for j=1:i-1
			res=res-O(:,j)*res*v(:,j);
		end
		O(:,i)=res/norm(res);
	end
	%Now we get X' matrix O，then count weight matrix W'
	if(hasOrder==1)
		W=[];
	else
		W=solveResult(O,Y);	
	end
	

end
function W=getW(X,Y)
	tol=10e-6;
	[m,n]=size(X);
	[a,b]=size(Y);
	tempX=zeros(m,m);
	tempY=zeros(m,n);
	for i=1:n
		tempX=X(:,i)*X(:,i)';
		tempY=Y(:,i)*X(:,i)';
	end
	[Q,R]=qr(tempX);
	tempY=tempY';
	R=R';
	solveResult=[];
	for i=1:m
		solveResult=[solveResult solveLowerTriangle(tempY(:,i),R,tol)];
	end
	solveResult=solveResult';
	W=solveResult*Q';
end
