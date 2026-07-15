%% step01_load_imu_bag.m  (모든 경로 상대경로)
% 목적: ../imu_analysis_bag 에서 '/imu/data'를 읽어 ../outputs/imu_raw.mat 저장

clear; clc;

%% [경로] 스크립트 기준 상대경로만 사용
here   = fileparts(mfilename('fullpath'));      % .../Git/imu_analysis/matlab
bagDir = fullfile(here, '..', 'imu_analysis_bag');
outDir = fullfile(here, '..', 'outputs');
if ~exist(bagDir, 'dir')
    error('rosbag2 폴더가 없습니다: %s\n폴더 구조를 확인하세요 (matlab와 imu_analysis_bag는 형제 폴더여야 함).', bagDir);
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

targetTopic = "/imu/data";   % 고정
fprintf('Bag folder (relative): %s\n', bagDir);
fprintf('Target topic: %s\n', targetTopic);

%% [열기] rosbag2 (버전 호환)
bag = [];
try
    bag = ros2bag(bagDir);                  % R2023b+
catch
    try
        bag = ros2bagreader(bagDir);        % 구버전
    catch ME
        error('ros2bag/ros2bagreader 로드 실패: %s', ME.message);
    end
end

%% [메타] 메시지 테이블에서 토픽 행 찾기 (상대경로 영향 없음)
props = string(properties(bag));
if any(props=="MessageList")
    T = bag.MessageList;
elseif any(props=="Messages")
    T = bag.Messages;
else
    error('메시지 테이블(MessageList/Messages)을 찾지 못했습니다.');
end
if ~istable(T) || ~any(strcmpi(T.Properties.VariableNames,'Topic'))
    error('메시지 테이블에 Topic 컬럼이 없습니다.');
end

% 안전 매칭(개행/공백 제거 후 비교)
topics_str  = string(T.Topic);
topics_norm = strip(erase(erase(topics_str, sprintf('\r')), sprintf('\n')));
rows = find(topics_norm == string(targetTopic));

if isempty(rows)
    disp(">> Bag에 있는 토픽 목록(상위 20개 표기):");
    disp( unique(topics_norm(1:min(20,end))) );
    error('토픽 %s 를 찾지 못했습니다. 위 목록을 확인하세요.', targetTopic);
end
fprintf('Matched rows: %d\n', numel(rows));

%% [읽기] readMessages (버전별 시그니처 호환)
try
    msgs = readMessages(bag, rows, "DataFormat", "struct");   % 신버전
catch
    tmp = readMessages(bag, rows);                             % 구버전
    if istable(tmp)
        msgs = table2struct(tmp);
    elseif iscell(tmp)
        if isempty(tmp), msgs = struct([]); 
        elseif isstruct(tmp{1}), msgs = [tmp{:}];
        else, error("readMessages 결과(셀 내부 타입) 미지원: %s", class(tmp{1}));
        end
    elseif isstruct(tmp)
        msgs = tmp;
    else
        error("readMessages 결과 타입 미지원: %s", class(tmp));
    end
end
if isempty(msgs), error('메시지를 읽지 못했습니다.'); end

%% [가공] 타임스탬프/샘플링/신호 추출
t  = arrayfun(@(m) double(m.header.stamp.sec) + double(m.header.stamp.nanosec)*1e-9, msgs);
t  = t - t(1);
dt = diff(t);         Fs = 1/median(dt);

ax = arrayfun(@(m) m.linear_acceleration.x, msgs);
ay = arrayfun(@(m) m.linear_acceleration.y, msgs);
az = arrayfun(@(m) m.linear_acceleration.z, msgs);
gx = arrayfun(@(m) m.angular_velocity.x,   msgs);
gy = arrayfun(@(m) m.angular_velocity.y,   msgs);
gz = arrayfun(@(m) m.angular_velocity.z,   msgs);
qw = arrayfun(@(m) m.orientation.w,        msgs);
qx = arrayfun(@(m) m.orientation.x,        msgs);
qy = arrayfun(@(m) m.orientation.y,        msgs);
qz = arrayfun(@(m) m.orientation.z,        msgs);

%% [저장] outputs도 상대경로
save(fullfile(outDir,'imu_raw.mat'), ...
     't','Fs','ax','ay','az','gx','gy','gz','qw','qx','qy','qz','targetTopic','-v7.3');

fprintf('Saved: %s\n', fullfile(outDir,'imu_raw.mat'));
fprintf('Topic: %s | Samples: %d | Duration: %.3fs | Fs~: %.2f Hz\n', targetTopic, numel(t), t(end), Fs);
