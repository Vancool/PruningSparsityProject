%solve the result of lower triangle matrix
function result=solveLowerTriangle(y,C,tol)
	result(1)=y(1)/C(1,1);
	m=length(y);
	for i=1:m
		k=0;
		if(i>=2)
			for j=1:i-1
				k=k+C(i,j)*result(j);
			end	
		end
		k=y(i)-k;
		if(C(i,i)<tol)
			result(i)=0;
		else
			result(i)=k/C(i,i);
        end
	end

end