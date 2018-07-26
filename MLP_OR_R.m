function [E,validIndexMatrix,vNum,w_in_hidden,w_fully_connected,bestLambda]=MLP_OR_R(X,Y,isSortedTask,vX,vY,hiddenNum,iterNum,prunRate,lambdaArray)
	lambdaSize=length(lambdaArray);
	lambda=lambdaArray(1);
	[inputNum,dataSize]=size(X);
	[outputNum,dataSize1]=size(Y);
	X=[X;ones(1,dataSize)];
	inputNum=inputNum+1;
	if dataSize~=dataSize1
		disp('data is invalid!');
		return 	
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
	%if is sorted task,prepare the target output 
	a=[];
	d=[];
	if(isSortedTask)
		Y0=ones(outputNum,dataSize)*(-0.5);
		for i=1:dataSize
			k=find(Y(:,i)==max(Y(:,i)));
			Y0(k,i)=0.5;
		end
		Y=Y0;
		a=zeros(1,dataSize);
		d=zeros(outputNum,dataSize);
	end
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
		loss=sum(sum(lossMatrix))/dataSize;
		countW=sum(sum(w_in_hidden.^2))+sum(sum(w_fully_connected.^2));
		loss=loss+lambda*countW;
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
			[oX,oW]=schimidtFun(xaMatrix(1:hiddenNum,:),Y,0);
			p=sortedInput(oW);
			[validIndexMatrix,vE,vNum]=MinimizeVError(oW,vX,vY,P(2,:));
		end
		%validIndexMatrix=repmat(validIndexMatrix,1,dataSize);
		%cut xa unit way function in w_clear_Matrix
		%X=X.*validIndexMatrix;
		%validIndexMatrix=validIndexMatrix(:,1);
		%update the weight and bias
		w_in_hidden=w_in_hidden-delta_weight_in_hidden;
		w_fully_connected=W0;
		%cut weight--->0
		w_clear_Matrix=repmat(validIndexMatrix,1,outputNum);
		w_clear_Matrix=w_clear_Matrix';
		w_hidden_out=w_fully_connected(:,1:hiddenNum);
		w_hidden_out=w_clear_Matrix.*w_hidden_out;
		w_fully_connected(:,1:hiddenNum)=w_hidden_out;
		w_clear_Matrix=repmat(validIndexMatrix,1,inputNum);
		w_in_hidden=w_clear_Matrix.*w_in_hidden;
		if(isSortedTask)
			a=sum(outputMatrix-Y-d)/outputNum;
			a=repmat(a,outputNum,1);
			d=outputMatrix-Y-a;
			%update the Y
			for i=1:dataSize
				tempY=Y(:,i);
				labelY=find(tempY==max(tempY));
				for j=1:outputNum
					if(j==labelY)
						if(d(j,i)>=0)
							Y(j,i)=Y(j,i)+a(j,i)+d(j,i);
						end
					else
						if(d(j,i)<=0)
							Y(j,i)=Y(j,i)+a(j,i)+d(j,i);
						end
					end
				end
			end
		end
	end
	%R procedure
	R0=zeros(fullConnectedNum,fullConnectedNum);
	for i=1:dataSize
		R0=R0+xa(:,i)*xa(:,i)';
	end
	r=[];
	for i=1:fullConnectedNum
		r=[r R0(i,i)];
	end
	r(end)=0;
	r=diag(r);
	lambda_error=[];
	W0_storage=cell(lambdaSize,1);
	for i=1:lambdaSize
		R=R0+lambdaArray(i)*r;
		tempResult=[];
		for i=1:outputNum
			temp=sovleLowerTriangle(C0(:,i),R,tol);
			tempResult=[tempResult temp];
		end
		tempResult=tempResult';
		W0=tempResult*Q';
		W0_storage(i)=W0;
		lambda_outputMatrix=W0*xaMatrix;
		loss=sum(sum((Y-lambda_outputMatrix).^2))/dataSize;
		countW=sum(sum(W0.^2))*lambdaArray(i);
		loss=loss+countW;
		lambda_error=[lambda_error loss];
	end
	lambda_index=find(lambda_error==min(lambda_error));
	bestLambda=lambdaArray(lambda_index);
	w_fully_connected=W0_storage(lambda_index);
	figure(1)
	title('lambda-Error graph')
	plot(lambdaArray,lambda_error,'*r','MarkerSize',15);
	xlabel('Lambda');
	ylabel('Error');


	figure(2)
	title('training epoch-error graph')
	plot(updateStep,lossValueArr,'LineWidth',3);
	xlabel('updating epoch');
	ylabel('MSE-Error');
end