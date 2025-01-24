clc;clear;

fileID=fopen('signal3.bin','r');
read_data=fread(fileID,4e6,'float');
fclose(fileID);

x=read_data(1:2:end)+1i*read_data(2:2:end);

%% For Signal and Signal2
pow=1;
PLL_Alpha=10;
PLL_Betta=20;
x=x/std(x);

FrameLen=1.6e6;
PLL_Out = PLL_V3_Enhanced_FB(x, pow , PLL_Alpha , PLL_Betta,FrameLen) ;

scatterplot(PLL_Out)




