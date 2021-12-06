clc;clear;
n_Sym_True=10e6;
modOrder       = 4;
Symbol_idx     = randi(modOrder,n_Sym_True,1);
Samples_QPSK   = exp(1j*(2*pi*(0:3)/4+pi/4)).';
Symbols        = Samples_QPSK(Symbol_idx);
output2=Symbols+(randn(size(Symbols))+1i*randn(size(Symbols)))/sqrt(2)*0.01;
% scatterplot(output2)

output=repmat(output2,10,1);
output=output.*exp(1i*pi/16+1*1i*pi/100*(1:numel(output))');
scatterplot(output)
saveData=[real(output),imag(output)].';
saveData=saveData(:);

fileID = fopen('signal3.bin','w');
fwrite(fileID,saveData,'float') ;
fclose(fileID);




