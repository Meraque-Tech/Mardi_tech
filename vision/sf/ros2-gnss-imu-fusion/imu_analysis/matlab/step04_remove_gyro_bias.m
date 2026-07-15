%% step04_remove_gyro_bias.m
% 목적: step03에서 구한 gyro bias를 제거한 데이터를 저장(+간단 비교 플롯)
clear; clc; close all;

here    = fileparts(mfilename('fullpath'));
outDir  = fullfile(here, '..', 'outputs');
rawFile = fullfile(outDir, 'imu_raw.mat');
biasFile= fullfile(outDir, 'gyro_bias.mat');
if ~exist(rawFile,'file'), error('imu_raw.mat 없음. step01 먼저.'); end
if ~exist(biasFile,'file'), error('gyro_bias.mat 없음. step03 먼저.'); end
S  = load(rawFile);
B  = load(biasFile);

t=S.t(:); Fs=S.Fs;
gx=S.gx(:); gy=S.gy(:); gz=S.gz(:);
% bias 제거
gx_c = gx - B.bx; gy_c = gy - B.by; gz_c = gz - B.bz;

% 저장
save(fullfile(outDir,'imu_bias_corrected.mat'),...
    't','Fs','gx','gy','gz','gx_c','gy_c','gz_c','-v7.3');

% 간단 플롯
figure('Name','Gyro bias removal');
subplot(3,1,1); plot(t,gx,t,gx_c); grid on; ylabel('gx (rad/s)'); title('Gyro bias removal'); legend('raw','corr');
subplot(3,1,2); plot(t,gy,t,gy_c); grid on; ylabel('gy (rad/s)'); legend('raw','corr');
subplot(3,1,3); plot(t,gz,t,gz_c); grid on; ylabel('gz (rad/s)'); xlabel('Time (s)'); legend('raw','corr');
saveas(gcf, fullfile(outDir,'04_gyro_bias_removal.png'));

fprintf('Saved: %s\n', fullfile(outDir,'imu_bias_corrected.mat'));
