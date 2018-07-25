function [E,validIndexMatrix,vNum,w_in_hidden,w_fully_connected]=MLP_P(X,Y,isAutoEncoder,vX,vY,hiddenNum,iterNum,prunRate)
	[inputNum,dataSize]=size(X);
	[outputNum,dataSize1]=size(Y);
	X=[X;ones(1,dataSize)];
	inputNum=inputNum+1;
	if dataSize~=dataSize1
		disp('data is invalid!');
		return 0	
	end
	learningRate=0.001;
	tol=10e-6;
	fullConnectedNum=inputNum+hiddenNum;
	w_in_hidden=randn(hiddenNum,inputNum);
	w_fully_connected=randn(outputNum,fullConnectedNum);
	if isAutoEncoder==1
		Y=X(1:inputNum-1,:);
	end
	number=0;
	lossValueArr=[];
	updateStep=[];
	pruneCount=0;
	while number<iterNum
		number=number+1;
		pruneCount=pruneCount+1;
		hiddenMatrix_before=w_in_hidden*X;
		hiddenMinMatrix=min(hiddenMatrix_before);
		hiddenMaxMatrix=max(hiddenMatrix_before);
		normMatrix=hiddenMaxMatrix-hiddenMinMatrix;
		normMatrix=repmat(normMatrix,hiddenNum,1);
		hiddenMinMatrix=repmat(hiddenMinMatrix,hiddenNum,1);
		hiddenMatrix_before=(hiddenMatrix_before-hiddenMinMatrix)./normMatrix;
		hiddenmatrix_after=sigmoid(hiddenMatrix_before);
		xaMatrix=[hiddenmatrix_after;X];
		outputMatrix=w_fully_connected*xaMatrix;
		loss=0;
		lossMatrix=(Y-outputMatrix).^2;
		loss=sum(sum(lossMatrix))*dataSize
		lossValueArr=[lossValueArr loss];
		E=loss;
		updateStep=[updateStep number];
		if(loss<0.1)
			%loss small enough
			break;
		end
		%update the weight first
		%OWO update w_fully_connected
		R0=size(fullConnectedNum,fullConnectedNum);
		C0=size(outputNum,fullConnectedNum);
		for i=1:dataSize
			R0=R0+xaMatrix(:,i)*xaMatrix(:,i)';
			C0=C0+Y(:,i)*xaMatrix(:,i)';
		end
		[Q,R]=qr(R0);
		R=R';
		C0=C0';
		tempResult=[];
		for i=1:outputNum
			temp=sovleLowerTriangle(C0(:,i),R,tol);
			tempResult=[tempResult temp];
		end
		tempResult=tempResult';
		W0=tempResult*Q';
		%END OWO
		%begin HWO find the best descend direction
		w_hidden_out=w_fully_connected(:,1:hiddenNum);
		G=JacobiW(outputMatrix,Y,hiddenmatrix_after,X,normMatrix,w_hidden_out);
		R1=zero(inputNum,inputNum);
		for i=1:inputNum
			R1=R1+X(:,i)*X(:,i)';
		end
		Ghwo=HWO(R1,G,tol);
		%END HWO
		z=ones(hiddenNum,1)*learningRate;
		z=repmat(z,1,inputNum);
		delta_weight_in_hidden=z.*Ghwo;
		%END MOLF

		%pruning using SFS
		if(number==prunRate)
			number=0;
			[oX,oW]=schimidtFun(xaMatrix,Y,0);
			p=sortedInput(oW);
			[validIndexMatrix,vE,vNum]=MinimizeVError(oW,vX,vY,P(2,:));
		end
		validIndexMatrix=repmat(validIndexMatrix,1,dataSize);
		%cut xa unit
		X=X.*validIndexMatrix;
		validIndexMatrix=validIndexMatrix(:,1);
		%update the weight and bias
		w_in_hidden=w_in_hidden-delta_weight_in_hidden;
		w_fully_connected=W0;
		%cut weight--->0
		w_clear_Matrix=repmat(validIndexMatrix(hiddenNum+1:end),1,hiddenNum);
		w_clear_Matrix=w_clear_Matrix';
		w_in_hidden=w_clear_Matrix.*w_in_hidden;
		w_clear_Matrix=repmat(validIndexMatrix,1,outputNum);
		w_fully_connected=w_clear_Matrix'.*w_fully_connected;
	end
	plot(updateStep,lossValueArr,'LineWidth',3);
	xlabel('updating epoch');
	ylabel('MSE-Error');
end