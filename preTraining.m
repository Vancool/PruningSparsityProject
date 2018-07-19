%pre-training function to pruning less useful inputs
function [inputValueVector,inputSortedList,W,W_Matrix,final_E]=preTraining(X,Y)
	%using gpu to training
	[m,trainSize]=size(X);
	%count correlation matrices
	R=zeros(m,m);
	for i=1:trainSize
		R=R+X(:,i)*X(:,i)';
	end
	[Q,A]=qr()
end