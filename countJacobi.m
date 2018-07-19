function [resultZ,jacobiMatrix,hessianMatrix]=countJacobi(X,w_f,w_hidden_output,G,hiddenNum,previousZ)
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