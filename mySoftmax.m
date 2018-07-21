function output=mySoftmax(x)
	output=exp(x)./sum(exp(x));
	[m,n]=size(output);
	for i=1:n
		for j=1:m
			if(isnan(output(j,i)))
				output(j,i)=1;
			end
		end
	end
end