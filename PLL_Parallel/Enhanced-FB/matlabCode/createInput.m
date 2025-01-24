clc;clear;close all;
% Generate QPSK Symbols
M = 4; % Modulation order for QPSK
nsymb = 3e6;
dataSymbols = randi([0 M-1],nsymb , 1); % Random transmitted symbols
txSignal = qammod(dataSymbols, M, 'UnitAveragePower', true); % Ideal QPSK constellation
% Add Noise to the Signal
SNR_actual = 18; % Actual SNR in dB
rxSignal = awgn(txSignal, SNR_actual, 'measured'); % Received signal with noise
output=rxSignal.*exp(1i*pi/16+1*1i*pi/100*(1:numel(rxSignal))');
% scatterplot(output)

saveData=[real(output),imag(output)].';
saveData=saveData(:);

fileID = fopen('signal3.bin','w');
fwrite(fileID,saveData,'float') ;
fclose(fileID);




