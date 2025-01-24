
function PLL_output = PLL_V3_Enhanced_FB(PLL_In , pow , PLL_Alpha , PLL_Betta,FrameLen)
%% framing - PLL
numFrames = 2^8;
subFrameLen = FrameLen / numFrames;
frames = zeros(subFrameLen+1, numFrames);
for i = 1:numFrames
    index = ((i - 1) * subFrameLen + 1):(i * subFrameLen+1);
    frames(:, i) = PLL_In(index);
end
%% processing each frame independently - PLL
PLL_Out = zeros(subFrameLen+1, numFrames);
nIteration = 16;
phase_mismatch = zeros(numFrames, 2); % [Phi applied to first sample, Phi applied to last sample]
for i = 1:numFrames
    [PLL_Out(: ,i), phase_mismatch(i, 1), phase_mismatch(i, 2)] = PLL_V3_EFB(frames(:, i) , pow , PLL_Alpha , PLL_Betta,nIteration);
end

misang = zeros(numFrames - 1, 1); % mismatch detection between phase of two frame
for i = 2: length(phase_mismatch)
    misang(i) =  phase_mismatch(i, 1) - phase_mismatch(i - 1, 2);
    misang(i) = round(4 * misang(i)) / 4;
end
misang = cumsum(misang);
for i = 2:length(phase_mismatch)
    PLL_Out(: ,i) = PLL_Out(: ,i) * exp(1i * 2*pi * -misang(i));
end
PLL_Out(end,:)=[];
%% Output reconstruction - PLL
PLL_output = PLL_Out(:);

figure(2)
scatterplot(PLL_output(100:end))


