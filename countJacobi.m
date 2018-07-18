function result=countJacobi(X,w_f,w_hidden_output,G,hiddenNum)
	Z=sym('Z',[1 hiddenNum]);
	Op=(w_hidden_output+G*)*X;


end