function output=softmaxDerivative(x,i,j)
	if(i==j)
		output=x(i)*(1-x(i));
	else
		output=x(i)*x(j);
	end

end