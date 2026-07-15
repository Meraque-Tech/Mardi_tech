%% step03_estimate_gyro_bias.m
% 목적: 히스테리시스 기반 정지 구간 검출 → 자이로 바이어스(bx,by,bz) 추정
% 출력:
%   ../outputs/gyro_bias.mat
%   ../outputs/03_static_window.png
%   ../outputs/03_gyro_bias_hist.png
%   ../outputs/03_gyro_bias_report.txt

clear; clc; close all;

%% [경로]
here    = fileparts(mfilename('fullpath'));
outDir  = fullfile(here, '..', 'outputs');
inFile  = fullfile(outDir, 'imu_raw.mat');
if ~exist(inFile,'file'); error('imu_raw.mat이 없습니다. 먼저 step01_load_imu_bag.m 실행'); end
if ~exist(outDir,'dir');  mkdir(outDir); end

%% [로드] (열 벡터 통일)
S  = load(inFile);
t  = S.t(:);        Fs = S.Fs;
gx = S.gx(:); gy = S.gy(:); gz = S.gz(:);
ax = S.ax(:); ay = S.ay(:); az = S.az(:);

%% [파라미터]
g_val        = 9.80665;
th_gyr       = 0.03;   % rad/s (사용자 지정)
th_acc_low   = 0.50;   % m/s^2  정지 '진입' 기준
th_acc_high  = 0.70;   % m/s^2  정지 '유지/이탈' 상한(스파이크 허용)
win_sec      = 1.0;    % s  이동평균 윈도우
gap_tol_sec  = 0.5;    % s  짧은 끊김 메우기
min_len_sec  = 5.0;    % s  최소 정지 길이
max_len_sec  = 20.0;   % s  최대 정지 길이(상한 캡)

Nw      = max(1, round(Fs*win_sec));
gap_tol = max(1, round(Fs*gap_tol_sec));
min_len = max(1, round(Fs*min_len_sec));
max_len = max(1, round(Fs*max_len_sec));

%% [지표] |gyro|, ||a|-g|
gyro_norm = sqrt(gx.^2 + gy.^2 + gz.^2);
acc_err   = abs(sqrt(ax.^2 + ay.^2 + az.^2) - g_val);

gyro_s = movmean(gyro_norm, Nw, 'Endpoints','shrink');
acc_s  = movmean(acc_err,   Nw, 'Endpoints','shrink');

%% [히스테리시스 정지 마스크]
is_static = false(numel(t),1);
for i = 2:numel(t)
    if ~is_static(i-1)
        % 진입: 둘 다 낮아야 함
        is_static(i) = (gyro_s(i) < th_gyr) && (acc_s(i) < th_acc_low);
    else
        % 유지: 자이로 낮고, 가속도는 상한 이내면 유지
        is_static(i) = (gyro_s(i) < th_gyr) && (acc_s(i) <= th_acc_high);
    end
end
is_static_raw = is_static;

%% [스파이크 보정] 짧은 0 구간 메우기
d = diff([false; is_static; false]);
z0 = find(d==-1); z1 = find(d==1)-1;  % 0 구간 [z0..z1]
for k = 1:numel(z0)
    if z1(k) >= z0(k) && (z1(k)-z0(k)+1) <= gap_tol
        is_static(z0(k):z1(k)) = true;
    end
end

%% [최소 길이] min_len 미만 구간 제거
d2 = diff([false; is_static; false]);
s1 = find(d2==1); e1 = find(d2==-1)-1; L1 = e1 - s1 + 1;
keep = L1 >= min_len;
mask_min = false(size(is_static));
for k = find(keep).'
    mask_min(s1(k):e1(k)) = true;
end
is_static = mask_min;

%% [최대 길이 캡] 가장 긴 구간 선택 후 상한 적용
d3 = diff([false; is_static; false]);
s2 = find(d3==1); e2 = find(d3==-1)-1; L2 = e2 - s2 + 1;

if isempty(L2)
    warning('정지 구간이 검출되지 않았습니다. 파라미터(th_acc_low/high, th_gyr) 조정 필요.');
    static_mask = false(size(is_static));
else
    [~,imax] = max(L2);
    i0 = s2(imax); i1 = e2(imax);
    if (i1 - i0 + 1) > max_len
        i1 = i0 + max_len - 1;
    end
    static_mask = false(size(is_static));
    static_mask(i0:i1) = true;
end

Ns = nnz(static_mask);
if Ns < min_len
    warning('충분히 긴 정지 구간이 없습니다 (N=%d, %.2fs). 파라미터 조정 권장.', Ns, Ns/Fs);
end

%% [바이어스 추정]
bx = mean(gx(static_mask), 'omitnan');
by = mean(gy(static_mask), 'omitnan');
bz = mean(gz(static_mask), 'omitnan');

sx = std(gx(static_mask), 0, 'omitnan');
sy = std(gy(static_mask), 0, 'omitnan');
sz = std(gz(static_mask), 0, 'omitnan');

%% [리포트 저장]
rep = fullfile(outDir,'03_gyro_bias_report.txt');
fid = fopen(rep,'w');
fprintf(fid, "Gyro bias estimate (rad/s)\n");
fprintf(fid, "----------------------------------\n");
fprintf(fid, "bx = %.6f   (std=%.6f)\n", bx, sx);
fprintf(fid, "by = %.6f   (std=%.6f)\n", by, sy);
fprintf(fid, "bz = %.6f   (std=%.6f)\n", bz, sz);
fprintf(fid, "\nSamples used: %d / %d (%.1f%%)  ~ %.2f s\n", Ns, numel(t), 100*Ns/numel(t), Ns/Fs);
fprintf(fid, "Fs = %.2f Hz\n", Fs);
fprintf(fid, "Params: win=%.2fs, th_gyr=%.3f, th_acc_low=%.2f, th_acc_high=%.2f, gap_tol=%.2fs, min_len=%.2fs, max_len=%.2fs\n", ...
        win_sec, th_gyr, th_acc_low, th_acc_high, gap_tol_sec, min_len_sec, max_len_sec);
fclose(fid);
fprintf('Saved report: %s\n', rep);

%% [그림1] 정지 신호/마스크
f1 = figure('Name','Static window detection');
subplot(3,1,1);
plot(t, gyro_norm); hold on; yline(th_gyr,'--'); grid on;
ylabel('|gyro| (rad/s)'); title('Static detection signals');
legend('|gyro|','th\_gyr','Location','northwest');
subplot(3,1,2);
plot(t, acc_err); hold on; yline(th_acc_low,'--'); yline(th_acc_high,'--'); grid on;
ylabel('||a|-g| (m/s^2)');
legend('| |a|-g |','th\_acc\_low','th\_acc\_high','Location','northwest');
subplot(3,1,3);
stairs(t, is_static_raw, 'Color',[.6 .6 .6], 'DisplayName','raw static'); hold on;
stairs(t, static_mask, 'k','LineWidth',1.5,'DisplayName','final static');
ylim([-0.1 1.1]); grid on; ylabel('static'); xlabel('Time (s)');
legend('Location','southwest');
saveas(f1, fullfile(outDir,'03_static_window.png'));

%% [그림2] 바이어스 히스토그램
f2 = figure('Name','Gyro bias histogram');
subplot(3,1,1); histogram(gx(static_mask), 50); grid on;
xlabel('gx (rad/s)'); ylabel('count'); title(sprintf('bx=%.6f, std=%.6f',bx,sx));
subplot(3,1,2); histogram(gy(static_mask), 50); grid on;
xlabel('gy (rad/s)'); ylabel('count'); title(sprintf('by=%.6f, std=%.6f',by,sy));
subplot(3,1,3); histogram(gz(static_mask), 50); grid on;
xlabel('gz (rad/s)'); ylabel('count'); title(sprintf('bz=%.6f, std=%.6f',bz,sz));
saveas(f2, fullfile(outDir,'03_gyro_bias_hist.png'));

%% [저장]
save(fullfile(outDir,'gyro_bias.mat'), ...
     'bx','by','bz','sx','sy','sz','Ns','Fs', ...
     'th_gyr','th_acc_low','th_acc_high','win_sec','gap_tol_sec','min_len_sec','max_len_sec','static_mask','-v7.3');

fprintf('Saved: %s\n', fullfile(outDir,'gyro_bias.mat'));
fprintf('Bias [rad/s]: bx=%.6f by=%.6f bz=%.6f  (N=%d, ~%.2fs)\n', bx, by, bz, Ns, Ns/Fs);
