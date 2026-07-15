%% step02_qc_overview.m
% 목적: imu_raw.mat 데이터를 불러와 성분별로 개별 플롯 (subplot)으로 시각화

clear; clc; close all;

%% [경로] 상대경로
here    = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, '..', 'outputs');
dataFile = fullfile(dataDir, 'imu_raw.mat');

if ~exist(dataFile, 'file')
    error('imu_raw.mat이 없습니다. 먼저 step01_load_imu_bag.m을 실행하세요.');
end

%% [로드]
S = load(dataFile);
t = S.t;

ax = S.ax; ay = S.ay; az = S.az;
gx = S.gx; gy = S.gy; gz = S.gz;
qx = S.qx; qy = S.qy; qz = S.qz; qw = S.qw;

%% [플롯] 가속도
figure('Name','Linear Acceleration');
subplot(3,1,1);
plot(t, ax, 'r'); grid on;
ylabel('ax (m/s^2)');
title('Linear Acceleration');
subplot(3,1,2);
plot(t, ay, 'g'); grid on;
ylabel('ay (m/s^2)');
subplot(3,1,3);
plot(t, az, 'b'); grid on;
ylabel('az (m/s^2)');
xlabel('Time (s)');

%% [플롯] 각속도
figure('Name','Angular Velocity');
subplot(3,1,1);
plot(t, gx, 'r'); grid on;
ylabel('gx (rad/s)');
title('Angular Velocity');
subplot(3,1,2);
plot(t, gy, 'g'); grid on;
ylabel('gy (rad/s)');
subplot(3,1,3);
plot(t, gz, 'b'); grid on;
ylabel('gz (rad/s)');
xlabel('Time (s)');

%% [플롯] 쿼터니언
figure('Name','Orientation (Quaternion)');
subplot(4,1,1);
plot(t, qx, 'r'); grid on;
ylabel('qx');
title('Orientation (Quaternion)');
subplot(4,1,2);
plot(t, qy, 'g'); grid on;
ylabel('qy');
subplot(4,1,3);
plot(t, qz, 'b'); grid on;
ylabel('qz');
subplot(4,1,4);
plot(t, qw, 'k'); grid on;
ylabel('qw');
xlabel('Time (s)');

disp('✅ QC subplot plots created');
