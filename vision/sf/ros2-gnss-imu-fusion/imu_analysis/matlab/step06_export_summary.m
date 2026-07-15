%% step06_export_summary.m
% 목적: 분석 결과 요약 리포트/CSV/그림 출력
% 입력: ../outputs/imu_raw.mat, ../outputs/imu_acc_lpf.mat
% 선택: ../outputs/gyro_bias.mat (바이어스/정지마스크 요약)
% 출력:
%   ../outputs/06_summary.csv
%   ../outputs/06_summary.txt
%   ../outputs/06_rms_bar.png
%   콘솔 요약

clear; clc; close all;

%% [경로]
here    = fileparts(mfilename('fullpath'));
outDir  = fullfile(here, '..', 'outputs');
rawF    = fullfile(outDir, 'imu_raw.mat');
lpfF    = fullfile(outDir, 'imu_acc_lpf.mat');   % step05 결과
biasF   = fullfile(outDir, 'gyro_bias.mat');     % step03 결과(선택)

if ~exist(rawF,'file')
    error('imu_raw.mat이 없습니다. 먼저 step01_load_imu_bag.m을 실행하세요. (%s)', rawF);
end
if ~exist(lpfF,'file')
    error('imu_acc_lpf.mat이 없습니다. 먼저 step05_acc_fft.m을 실행하세요. (%s)', lpfF);
end
if ~exist(outDir,'dir'), mkdir(outDir); end

%% [로드] 원본/LPF/바이어스
S   = load(rawF);      % t, Fs, ax, ay, az, ...
P   = load(lpfF);      % imu_acc_lpf.*
hasBias = exist(biasF,'file');
if hasBias, B = load(biasF); end

% 원본
t  = S.t(:);  Fs = S.Fs;  N = numel(t);
ax = S.ax(:); ay = S.ay(:); az = S.az(:);

% LPF 결과(축별 fc 우선, 없으면 단일 fc 호환)
imu_acc_lpf = P.imu_acc_lpf;
if isfield(imu_acc_lpf,'ax_f')
    ax_f = imu_acc_lpf.ax_f(:); ay_f = imu_acc_lpf.ay_f(:); az_f = imu_acc_lpf.az_f(:);
else
    error('imu_acc_lpf.mat 내에 ax_f/ay_f/az_f가 없습니다.');
end
if isfield(imu_acc_lpf,'fc_x')
    fc_x = imu_acc_lpf.fc_x;  fc_y = imu_acc_lpf.fc_y;
    if isfield(imu_acc_lpf,'fc_z'), fc_z = imu_acc_lpf.fc_z; else, fc_z = imu_acc_lpf.fc_x; end
else
    if isfield(imu_acc_lpf,'fc')
        fc_x = imu_acc_lpf.fc; fc_y = imu_acc_lpf.fc; fc_z = imu_acc_lpf.fc;
    else
        error('imu_acc_lpf에 fc 정보가 없습니다.');
    end
end

% 정지 마스크(있으면 동적 구간 통계 제공)
dyn_mask = true(N,1);
if hasBias && isfield(B,'static_mask') && numel(B.static_mask)==N
    dyn_mask = ~B.static_mask(:);
end

%% [지표] RMS 전/후 & 개선율
rms_raw = @(x) sqrt(mean(x.^2,'omitnan'));
R_raw = [rms_raw(ax), rms_raw(ay), rms_raw(az)];
R_flt = [rms_raw(ax_f), rms_raw(ay_f), rms_raw(az_f)];
impr  = 100*(R_raw - R_flt)./max(R_raw, eps);   % 개선율(%)

% 동적구간만 별도 RMS (있으면 의미↑)
if any(~dyn_mask)
    R_raw_dyn = [rms_raw(ax(dyn_mask)), rms_raw(ay(dyn_mask)), rms_raw(az(dyn_mask))];
    R_flt_dyn = [rms_raw(ax_f(dyn_mask)), rms_raw(ay_f(dyn_mask)), rms_raw(az_f(dyn_mask))];
    impr_dyn  = 100*(R_raw_dyn - R_flt_dyn)./max(R_raw_dyn, eps);
else
    R_raw_dyn = nan(1,3); R_flt_dyn = nan(1,3); impr_dyn = nan(1,3);
end

%% [Welch PSD] > fc 고주파 전력 비율(전/후)
rx   = hf_ratio_for(ax,   fc_x, Fs);
rx_f = hf_ratio_for(ax_f, fc_x, Fs);
ry   = hf_ratio_for(ay,   fc_y, Fs);
ry_f = hf_ratio_for(ay_f, fc_y, Fs);
rz   = hf_ratio_for(az,   fc_z, Fs);
rz_f = hf_ratio_for(az_f, fc_z, Fs);

% 동적 구간 비율(있으면)
if any(~dyn_mask)
    rx_d  = hf_ratio_for(ax(dyn_mask),   fc_x, Fs);  rx_fd = hf_ratio_for(ax_f(dyn_mask),   fc_x, Fs);
    ry_d  = hf_ratio_for(ay(dyn_mask),   fc_y, Fs);  ry_fd = hf_ratio_for(ay_f(dyn_mask),   fc_y, Fs);
    rz_d  = hf_ratio_for(az(dyn_mask),   fc_z, Fs);  rz_fd = hf_ratio_for(az_f(dyn_mask),   fc_z, Fs);
else
    rx_d=nan; rx_fd=nan; ry_d=nan; ry_fd=nan; rz_d=nan; rz_fd=nan;
end

%% [CSV 저장]
axesNames = {'ax','ay','az'}';
fc_vec    = [fc_x; fc_y; fc_z];
T = table(axesNames, fc_vec, ...
          R_raw(:), R_flt(:), impr(:), ...
          [rx; ry; rz], [rx_f; ry_f; rz_f], ...
          'VariableNames', {'axis','fc_Hz','rms_raw','rms_filt','rms_improve_pct', ...
                            'HF_ratio_raw','HF_ratio_filt'});
writetable(T, fullfile(outDir,'06_summary.csv'));

%% [텍스트 요약 저장]
txtF = fullfile(outDir,'06_summary.txt');
fid = fopen(txtF,'w');
fprintf(fid, "=== IMU Processing Summary ===\n");
fprintf(fid, "Fs = %.2f Hz | N = %d | Duration = %.2f s\n\n", Fs, N, t(end));
if hasBias && isfield(B,'bx')
    Ns = getfield(B,'Ns'); %#ok<GFLD>
    fprintf(fid, "Gyro bias [rad/s]: bx=%.6f  by=%.6f  bz=%.6f  (static ~ %.2f s)\n\n", ...
        B.bx, B.by, B.bz, Ns/Fs);
end
fprintf(fid, "[LPF cutoff]\n");
fprintf(fid, "ax: %.2f Hz | ay: %.2f Hz | az: %.2f Hz\n\n", fc_x, fc_y, fc_z);

fprintf(fid, "[RMS (full)]\n");
fprintf(fid, "axis   raw      filt     improve(%%)\n");
for i=1:3
    fprintf(fid, "%-3s  %8.4f  %8.4f   %6.1f\n", axesNames{i}, R_raw(i), R_flt(i), impr(i));
end

fprintf(fid, "\n[HF ratio > fc (full)]\n");
fprintf(fid, "axis   raw(%%)  filt(%%)\n");
hf_raw  = 100 * [rx ry rz];
hf_filt = 100 * [rx_f ry_f rz_f];
for i=1:3
    fprintf(fid, "%-3s   %6.1f   %6.1f\n", axesNames{i}, hf_raw(i), hf_filt(i));
end

if any(~dyn_mask)
    fprintf(fid, "\n[Dynamic-only RMS]\n");
    for i=1:3
        fprintf(fid, "%-3s  %8.4f  %8.4f   %6.1f\n", axesNames{i}, R_raw_dyn(i), R_flt_dyn(i), impr_dyn(i));
    end
    fprintf(fid, "\n[Dynamic-only HF ratio > fc]\n");
    fprintf(fid, "ax   raw %.1f%%  → filt %.1f%%\n", 100*rx_d,  100*rx_fd);
    fprintf(fid, "ay   raw %.1f%%  → filt %.1f%%\n", 100*ry_d,  100*ry_fd);
    fprintf(fid, "az   raw %.1f%%  → filt %.1f%%\n", 100*rz_d,  100*rz_fd);
end
fclose(fid);

%% [그림] RMS 전/후 막대그래프
figure('Name','RMS improvement','NumberTitle','off');
vals = [R_raw(:) R_flt(:)];
bar(vals); grid on;
set(gca,'XTickLabel',{'ax','ay','az'});
legend('raw','filtered','Location','northoutside','Orientation','horizontal');
ylabel('RMS (m/s^2)'); title('RMS before/after LPF');
saveas(gcf, fullfile(outDir,'06_rms_bar.png'));

%% [콘솔 출력]
disp('--- SUMMARY (console) ---');
disp(T);
fprintf('Saved: %s\n', fullfile(outDir,'06_summary.csv'));
fprintf('Saved: %s\n', txtF);
fprintf('Saved: %s\n', fullfile(outDir,'06_rms_bar.png'));

%% ===== 로컬 함수 =====
function rr = hf_ratio_for(x, fc, Fs)
    % Welch PSD로 전체 전력과 >fc 전력 비율 계산
    nfft = 2^nextpow2(round(Fs*2));   % ~2초 창
    if nfft > numel(x), nfft = 2^nextpow2(max(8, floor(numel(x)/2))); end
    if nfft < 16, nfft = 16; end
    [Pxx,f] = pwelch(x, hamming(nfft), round(0.5*nfft), nfft, Fs, 'onesided');
    tot = trapz(f, Pxx);
    if tot <= 0, rr = 0; return; end
    idx = f >= fc;
    rr = trapz(f(idx), Pxx(idx)) / tot;
end
