%get label
file=dir('trainingDigits/*.txt');
TrainDataSize=length(file);
TrainDataLabel=zeros(TrainDataSize，1);
trainData=cell(TrainDataSize，1);
%USEING regular expression to get label
for i=1:TrainDataSize
	name=file(i).name
	ms=regexp(name,'[0-9]','match');
	TrainDataLabel(i)=str2num(ms{1});%ms is cell
	fid=fopen(['trainingDigits/',file(i).name],'r');
	traindata=fscanf(fid,'%s');
	data=zeros(32,32);
	
end
