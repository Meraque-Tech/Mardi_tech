%% step05_acc_fft.m
% 목적: 자동차용 — x,y 축 중심 FFT 분석(0.1~5 Hz)으로 축별 LPF(fc) 선정 후 zero-phase 필터링
% 입력: ../outputs/imu_raw.mat  (t, Fs, ax, ay, az)
% 선택: ../outputs/gyro_bias.mat (static_mask 있으면 정지 제외 구간만 FFT 분석)
% 출력: ../outputs/imu_acc_lpf.mat, 그림 2장

clear; clc; close all;

%% [경로]
here     = fileparts(mfilename('fullpath'));
outDir   = fullfile(here, '..', 'outputs');
rawFile  = fullfile(outDir, 'imu_raw.mat');
biasFile = fullfile(outDir, 'gyro_bias.mat');

if ~exist(rawFile,'file'); error('imu_raw.mat 없음: %s', rawFile); end
if ~exist(outDir,'dir'); mkdir(outDir); end

%% [로드]
S  = load(rawFile);       % t, Fs, ax, ay, az, ...
t  = S.t(:);  Fs = S.Fs;
ax = S.ax(:); ay = S.ay(:); az = S.az(:);
N  = numel(t);
fprintf('Loaded: N=%d, Fs=%.2fHz, T=%.1fs\n', N, Fs, t(end));

%% [분석 구간] 정지 제외(있으면)
use_mask = false; sel = true(N,1);
if exist(biasFile,'file')
    B = load(biasFile);
    if isfield(B,'static_mask') && numel(B.static_mask)==N
        dyn_mask = ~B.static_mask(:);
        if nnz(dyn_mask) > 0.2*N
            sel = dyn_mask; use_mask = true;
        end
    end
end
if use_mask
    fprintf('FFT는 동적 구간만 사용: %d samples (%.1fs)\n', nnz(sel), nnz(sel)/Fs);
else
    fprintf('FFT는 전체 구간 사용\n');
end

%% [FFT] 자동차 동역학 대역(0.1~5 Hz), x/y 우선
band_lo = 0.1; band_hi = 5.0;
sigX = detrend(ax(sel),0);
sigY = detrend(ay(sel),0);
sigZ = detrend(az(sel),0); % 참고용

M = numel(sigX);
f = (0:M-1)' * (Fs/M);
mask = (f>band_lo) & (f<band_hi);

AX = fft(sigX); AY = fft(sigY); AZ = fft(sigZ);
if ~any(mask); error('대역(%.1f~%.1f Hz)에 스펙트럼 없음. Fs/M 확인.', band_lo, band_hi); end
fb = f(mask);

[~,ix] = max(abs(AX(mask)));
[~,iy] = max(abs(AY(mask)));
[~,iz] = max(abs(AZ(mask)));  %#ok<ASGLU>  % z는 참고

f_vib_x = fb(ix);  f_vib_y = fb(iy);
fprintf('[FFT peaks] fx=%.2f Hz, fy=%.2f Hz  (band %.1f–%.1f)\n', f_vib_x, f_vib_y, band_lo, band_hi);

%% [축별 cut-off] 피크*1.2, 자동차용 클램프(0.5~5 Hz)
fc_x = min(max(1.2*f_vib_x, 0.5), min(5, Fs/4));
fc_y = min(max(1.2*f_vib_y, 0.5), min(5, Fs/4));
% z는 굳이 중요치 않지만 동일 규칙으로
fc_z = min(max(1.2*fb(iz),   0.5), min(5, Fs/4));
fprintf('[LPF fc] fx=%.2f Hz, fy=%.2f Hz (fz=%.2f Hz 참고)\n', fc_x, fc_y, fc_z);

%% [Zero-phase LPF] Butterworth 4차 + filtfilt
ord = 4;
[bx,axf] = butter(ord, fc_x/(Fs/2));
[by,ayf] = butter(ord, fc_y/(Fs/2));
[bz,azf] = butter(ord, fc_z/(Fs/2));
ax_f = filtfilt(bx,axf, ax);
ay_f = filtfilt(by,ayf, ay);
az_f = filtfilt(bz,azf, az);

%% [플롯] x/y/z FFT (선택 구간 기준)
fig1 = figure('Name','Accel FFT (x/y focus)','NumberTitle','off');
plot(f, abs(AX), 'r', f, abs(AY), 'g', f, abs(AZ), 'b'); hold on;
xline(f_vib_x,'--r','peak x'); xline(f_vib_y,'--g','peak y');
xlim([0 8]); grid on;
xlabel('Frequency (Hz)'); ylabel('|FFT|');
title(sprintf('Accel FFT (segment %ds–%ds)  band=%.1f–%.1f Hz', ...
      round(t(find(sel,1,'first'))), round(t(find(sel,1,'last'))), band_lo, band_hi));
legend('a_x','a_y','a_z','Location','northeast');
saveas(fig1, fullfile(outDir,'05_fft_xy.png'));

%% [플롯] 시간영역 비교 – 축별 fc 표기
fig2 = figure('Name','Accel LPF (axis-specific fc)','NumberTitle','off');
tiledlayout(3,1,"TileSpacing","compact","Padding","compact");

nexttile; plot(t, ax,'k--', t, ax_f,'r'); grid on;
ylabel('a_x (m/s^2)');
title(sprintf('a_x  raw(--) vs LPF(-)  fc=%.2f Hz', fc_x));

nexttile; plot(t, ay,'k--', t, ay_f,'g'); grid on;
ylabel('a_y (m/s^2)');
title(sprintf('a_y  raw(--) vs LPF(-)  fc=%.2f Hz', fc_y));

nexttile; plot(t, az,'k--', t, az_f,'b'); grid on;
ylabel('a_z (m/s^2)'); xlabel('Time (s)');
title(sprintf('a_z  raw(--) vs LPF(-)  fc=%.2f Hz (ref)', fc_z));
legend('raw','LPF','Location','best');
saveas(fig2, fullfile(outDir,'05_lpf_xy_timeseries.png'));

%% [저장]
imu_acc_lpf = struct('Fs',Fs, ...
    'fc_x',fc_x,'fc_y',fc_y,'fc_z',fc_z,'ord',ord, ...
    'ax',ax,'ay',ay,'az',az, ...
    'ax_f',ax_f,'ay_f',ay_f,'az_f',az_f, ...
    'band',[band_lo band_hi],'used_dyn_mask',use_mask);
save(fullfile(outDir,'imu_acc_lpf.mat'),'imu_acc_lpf','-v7.3');

fprintf('Saved: %s\n', fullfile(outDir,'imu_acc_lpf.mat'));
fprintf('Saved plots: %s, %s\n', ...
    fullfile(outDir,'05_fft_xy.png'), ...
    fullfile(outDir,'05_lpf_xy_timeseries.png'));
