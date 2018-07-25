%MOLF -count the jacobi matrix and hessian matrix of output error function respect to learning rate
function [resultZ,jacobiMatrix,hessianMatrix]=MOLF_Z(X,w_f,w_in_hidden,G,previousZ)
    [inputNum,dataSize]=size(X);
    [hiddenNum,inputNum]=size(w_in_hidden);
	Z=sym('Z',[hiddenNum,1]);
    
	temp=[];
	tempZ=repmat(Z,1,inputNum);
	temp=G.*tempZ;
	Op=(w_in_hidden+temp)*X;
	OpMin=min(Op);
	OpMax=max(Op);
	OpNorm=OpMax-OpMin;
	OpNorm=repmat(OpNorm,hiddenNum,1);
	OpMin=repmat(OpMin,hiddenNum,1);
	Op=(Op-OpMin)./OpNorm;
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