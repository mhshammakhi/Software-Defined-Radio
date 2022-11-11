clc;clear;close all;
fd=fopen('signal_Fc1_SPS4_ModuQPSK_SNR15_float.bin','r');
fseek(fd,1e6,'bof');
A=fread(fd,10e6,'float');
fclose(fd);
signal_int=A(1:2:end)+1i*A(2:2:end);
pwelch(signal_int(1:1e6),2048,[],[],'centered')

fd=fopen('BB_Out.bin','r');
A=fread(fd,10e6,'float');
fclose(fd);
signal_filt=A(1:1e6)+1i*A(1e6+1:end);
pwelch(signal_filt,2048,[],[],'centered')
fd=fopen('SDR_Output.bin','r');
A=fread(fd,10e6,'float');
fclose(fd);
signal_out=A(1:2:end)+1i*A(2:2:end);

scatterplot(signal_out(1e4:end));