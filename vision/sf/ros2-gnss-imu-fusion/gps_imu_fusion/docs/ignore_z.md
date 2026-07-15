# EKF에서 Z(고도/수직속도/수직가속도) 완전 무시: 2D 모드 설계안

목표: 주행체를 평평한 2D 평면에서만 추정하도록 구성한다. GPS의 고도(altitude)와 수직 속도, IMU의 수직 가속도(중력 보정 포함)를 융합에서 배제하고, 출력은 x, y, heading, vx, vy만 의미를 갖도록 한다. IMU는 차량 바닥과 평행하게 장착되어 있고 z축이 연직 방향을 본다고 가정한다(현 장착과 거의 일치).

## 현재 구현 요약 (2025-09)
- 상태공간: 15-state EKF (pos xyz, vel xyz, att roll/pitch/yaw, accel bias xyz, gyro bias xyz).
- 차량 평지 가정: roll, pitch는 0으로 고정하고 yaw만 추정/보정함.
- 입력/출력 요약:
  - GNSS: `NavSatFix`(위치), `TwistWithCovarianceStamped`(속도) 직접 구독 후 EKF에 전달. 레버암 보정 포함.
  - IMU: `/imu/processed` 구독. 전처리에서 편향 제거 및 LPF 적용. EKF에서는 yaw 축 중심으로 사용.
  - 출력: `nav_msgs/Odometry`(`odometry/filtered`), `PoseStamped`(`pose/filtered`), TF(odom→base_link), Path.

실제 코드 포인트(참고):
- GNSS 좌표/속도 EKF 업데이트: `INS/gps_imu_fusion/src/ekf_fusion_node.cpp`
- EKF 코어(상태/잡음/측정 H, R 등): `INS/gps_imu_fusion/src/kai_ekf_core.cpp`
- IMU 전처리: `INS/imu_preprocess/src/imu_preprocess_node.cpp`

## 외부 레퍼런스(일반 관행)
Robot Localization 계열에서 2D만 사용하려면 아래가 정석이다.
- EKF: `two_d_mode: true`로 z, roll, pitch를 드랍하고 yaw만 사용.
- NavSat Transform: `zero_altitude: true`로 고도를 0으로 강제, 수평 좌표만 퍼블리시.
- 센서 설정: IMU는 중력 제거, yaw만 유효, GPS z 공분산은 크게 설정하거나 사용 안 함.

본 패키지는 커스텀 EKF이므로 위 동작을 “동등하게” 구현하면 된다.

## 설계 선택지

### A안: 비침투적 2D 토글(권장, 빠른 적용)
EKF의 내부 차원(15-state)은 유지하되, Z 관련 입력/측정/출력을 무시하거나 평면화한다. 파라미터 한 개(`two_d_mode`)로 토글.

핵심 아이디어
- 입력 평면화: GNSS z, v_z, IMU a_z를 0으로 강제하거나 “측정 노이즈 무한대”로 무시 처리.
- 출력 평면화: Odom/TF의 z와 v_z를 0으로 고정. roll/pitch는 이미 0 유지.
- 공분산 처리: z 관련 분산은 매우 크게, z 연관 교차항은 0으로.

적용 포인트 (제안 변경)
1) 노드 파라미터 추가: `EkfFusionNode`
   - 이름: `two_d_mode` (bool, default: false)
   - 위치: `INS/gps_imu_fusion/src/ekf_fusion_node.cpp`의 `loadParameters()`에 선언/로드.

2) GNSS 위치 콜백(gnssCallback)
   - 현재: 위경도→UTM→로컬(x,y,z) 계산 후 레버암 보정, 다시 lat/lon/alt로 변환하여 EKF에 전달.
   - 2D 모드: 레버암 보정 이후 `base_local_z = 0.0`으로 강제, `coor.alt = reference_altitude_`로 고정.
   - GNSS 위치 공분산 전달 시 z 분산(`pos_cov[8]`)을 매우 크게 덮어쓰기(예: 1e12) 또는 그대로 두고 코어에서 무시.

3) GNSS 속도 콜백(gnssVelCallback)
   - 현재: vN, vE, vD(NED) 구성 후 EKF에 전달. 레버암 보정 포함.
   - 2D 모드: `vel.vD = 0.0` 강제. 속도 공분산 z(`vel_cov[8]`)도 매우 크게.

4) IMU 콜백(imuCallback)
   - 현재: IMU를 base_link로 회전만 적용 후(레버암 보정 포함) EKF에 전달. roll/pitch는 0으로 고정, yaw만 사용.
   - 2D 모드: `imu_data.accZ = 0.0`로 전달(수직 동역학 영향 제거). 이미 roll/pitch 각속도는 0 취급 중.
   - IMU 공분산 전달 시 z축 가속도/각속도 분산을 크게 하거나, 그대로 두고 코어에서 수직 Q를 축소/무시.

5) Odom/TF 퍼블리시
   - `publishOdometry()`와 `publishTransform()`에서 `position.z = 0.0`, `twist.linear.z = 0.0` 강제.
   - Odom 공분산에서 z 관련 분산(행렬 [2,2] 또는 ROS 6x6 인덱스 14)을 매우 크게(예: 1e12) 세팅, 교차항은 0.

6) EKF 코어(kai_ekf_core)
   - “빠른 적용”에서는 코드 변경 없이 파라미터만으로 z측정 무시 가능:
     - 측정 R 행렬의 z 관련 분산을 아주 크게: `gps_pos_noise_d`, `gps_vel_noise_d`를 큰 값으로 설정(아래 파라미터 오버레이 예시).
   - 안정성 보강(선택): 2D 모드일 때 프로세스 노이즈에서 z항 영향 축소/동결
     - `updateProcessNoiseMatrix()`에서 z 관련 항(`Rw(2,2)` 등)을 0 또는 매우 작게.
     - `updateJacobianMatrix()`에서 z 연동 항의 영향 최소화(현 구조에선 필수는 아님).

장점/단점
- 장점: 변경 폭이 작고 바로 테스트 가능. 기존 15-state와 호환.
- 단점: 내부적으로 z 상태는 존재하며 drift할 수 있으나, 출력/측정에 영향을 주지 않도록 가둠(무시).

### B안: EKF 내부 2D화(심층 리팩터링)
- 15-state를 2D 차량 친화 상태(예: x,y,yaw,vx,vy,gyro_bias_z, …)로 축소. H/F/Q/R 전면 수정.
- 장점: 계산/튜닝 단순화, z 드리프트 자체 제거.
- 단점: 공사 범위 큼. 단기간 적용 비권장. A안으로 충분히 검증된 후 고려.

## 파라미터 오버레이(빠른 2D 테스트용)
`INS/gps_imu_fusion/config/fusion_params.yaml`를 그대로 두고, 2D 전용 덮어쓰기 파일을 하나 추가해서 런치에 바인딩하는 방식이 안전하다. 예시:

```yaml
ekf_fusion_node:
  ros__parameters:
    two_d_mode: true

    # Z 측정 완전 무시를 위해 z 분산을 매우 크게
    gps_pos_noise_d: 1.0e6
    gps_vel_noise_d: 1.0e6

    # 출력 평면화(코드 반영 시 불필요할 수 있으나 안전하게 유지)
    publish_tf: true
```

런치에서 오버레이 적용 예:
```bash
ros2 launch gps_imu_fusion ekf_fusion.launch.py params_file:=/path/to/fusion_params_2d.yaml
```

추가로, 테스트 동안 센서 레버암 z 성분 영향을 제거하고 싶다면 `tf_static.launch.py` 인자에서 `gps_z`, `imu_z`를 0으로 주는 것도 방법이다(선택).

## 코드 변경 체크리스트(요약)
필수(최소 변경):
- EkfFusionNode에 `two_d_mode` 파라미터 추가 및 캐시.
- gnssCallback: `if (two_d_mode) base_local_z=0; coor.alt=reference_altitude_`.
- gnssVelCallback: `if (two_d_mode) vel.vD=0`.
- imuCallback: `if (two_d_mode) imu_data.accZ=0`.
- publishOdometry/Transform: `if (two_d_mode) z=0, v_z=0, z분산↑`.
- 파라미터 오버레이: `gps_pos_noise_d`, `gps_vel_noise_d` 매우 크게.

선택(안정성 강화):
- kai_ekf_core: 2D 모드 시 Rw의 z 관련 항 축소/동결, R의 z 항을 항상 매우 크게 유지.
- z 관련 교차 공분산을 0으로 정규화하는 가드 추가.

## 테스트 계획
1) 유닛: 
   - 무부하 실행(센서 토픽 echo)에서 `odometry/filtered.pose.pose.position.z`와 `twist.linear.z`가 항상 0인지 확인.
   - 공분산 z(odom.pose.covariance[14], twist.covariance[14])가 매우 큰 값으로 유지되는지 확인.
2) 주행 로그(rosbag):
   - 평지 정지/출발/선회/가감속 시 yaw와 (x,y) 추정의 안정성 확인.
   - ZUPT 전이 로직 영향(정지/전환/주행) 정상 동작 확인(기존과 동일해야 함).
3) 스트레스: 
   - GPS가 순간적으로 터지는(altitude 스파이크) 구간에서 출력 z=0 유지됨 확인.
   - IMU 수직 충격(방지턱 등) 상황에서 yaw/xy가 흔들리지 않는지 확인.

권장 시각화:
- `rviz2`에서 TF(odom→base_link)와 Path 확인: 경로가 z=0 평면에 붙어있어야 함.
- `rqt_plot`으로 `odometry/filtered/twist/twist/linear/z` 추적(항상 0).

## 리스크/주의사항
- 레버암 보정의 수직 성분을 무시하면, 큰 Z 오프셋에서 급회전 시 v_z 보정 항이 제거됨. 2D에서는 의도된 동작이지만, 고속 큰 롤/피치가 가능한 플랫폼에는 부적합.
- 내부 EKF z 상태는 여전히 존재하므로, 장시간 드리프트할 수 있다. 단, 측정/출력 경로에서 z를 무시하므로 결과에는 영향이 없어야 한다. 필요 시 코어에서 z 관련 Q를 더 축소해 동결.
- 다운스트림 노드가 z 분산을 의미있게 사용한다면(예: 3D 플래너), z 분산을 매우 크게 주는 대신 명시적으로 해당 축을 읽지 않도록 구성 필요.

## 결론
단기적으로는 A안(비침투 2D 토글)로 빠르게 적용하고, 주행 데이터로 성능/안정성을 확인한 뒤 필요하면 B안(내부 2D화)을 검토한다. 위 체크리스트와 파라미터 오버레이만으로도 “z 완전 무시” 요구사항을 충족하며, 기존 코드 영향 범위가 작아 리스크가 낮다.
