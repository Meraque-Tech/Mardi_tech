# GPS-IMU Fusion 패키지 분석 보고서 (2025.08 최신)

## 1. 현재 구현 상태

### 1.1 핵심 기능 (구현 완료)
- **15-state EKF**: 위치(3), 속도(3), 자세(3), 가속도 bias(3), 자이로 bias(3)
- **ZUPT (Zero Velocity Update)**: 3단계 속도 기반 히스테리시스 적용
- **GPS Heading 융합**: 속도별 동적 노이즈 조절로 GPS/IMU 균형 융합
- **좌표계 변환**: UTM 변환 및 로컬 카르테시안 좌표계 구현
- **스레드 안전성**: shared_mutex로 멀티스레드 환경 대응

### 1.2 IMU Calibration 현황
- 자이로 bias만 EKF에서 온라인 추정
- 가속도계 bias는 전처리 단계에서 처리 가정
- Allan variance, scale factor 등 고급 calibration 미구현

## 2. ZUPT 및 GPS Heading 융합 구현

### 2.1 3단계 속도 기반 융합 전략 (구현 완료)
```
┌─────────────┬──────────────┬────────────────┬──────────────────┐
│    상태      │  속도 범위     │  GPS Heading   │   융합 전략        │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ 1. 정지      │ < 0.08 m/s   │ OFF (무시)      │ IMU only         │
│             │              │ noise = 1000   │ Q scale = 0.1    │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ 2. 전환      │ 0.08~0.5 m/s │ 극히 높은 가중치  │ GPS heading 우선  │
│   (저속)     │              │ noise = 0.001  │ Q scale = 1.0    │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ 3. 주행      │ > 0.5 m/s    │ 보통 가중치      │ GPS/IMU 균형      │
│   (고속)     │              │ noise = 0.1    │ Q scale = 1.0    │
└─────────────┴──────────────┴────────────────┴──────────────────┘
```

### 2.2 히스테리시스 적용
- 정지→이동 전환: 0.12 m/s 이상, 0.3초 유지
- 이동→정지 전환: 0.08 m/s 이하, 0.3초 유지
- 전환 상태 최소 유지: 0.4초 (GPS heading 보정 시간 확보)
- IMU 정지 감지와 OR 조건으로 결합

## 3. 좌표계 및 TF 변환 구조

### 3.1 현재 구현
- **TF 발행**: odom → base_link (ekf_fusion_node.cpp:786-813)
- **원점**: 건국대 일감호 고정 (37.540091, 127.076555)
- **좌표 변환**: GPS → UTM → 로컬 카르테시안
- **고도 처리**: `z = ekf_alt_m - reference_altitude_`

### 3.2 Lever Arm 보정 ✅ 완료 (2025.08.27)
- **GPS 안테나**: base_link에서 (0.5m, 0, 0.2m) 오프셋 → TF로 자동 보정
- **IMU**: base_link에서 (-0.3m, 0, 0) 오프셋 → TF로 자동 보정
- **구현 완료**: GPS position/velocity, IMU acceleration에 대한 lever arm dynamics 보정

## 4. 해결된 문제들

### 4.1 Critical 이슈 (✅ 해결)
- **스레드 안전성**: shared_mutex 올바르게 구현
- **비정상 dt 처리**: dt ≤ 0 시 스킵
- **수치적 안정성**: asin() 범위 제한, 쿼터니언 NaN 체크
- **ZUPT 구현**: 3단계 속도 기반 융합 완료
- **GPS Heading 융합**: 동적 노이즈 조절 구현
- **Lever Arm 보정**: GPS/IMU lever arm dynamics 완전 구현 (2025.08.27)

### 4.2 프레임/축 정의 수정 (✅ 해결, 2025.09.05)
- **GNSS 속도 프레임 매핑 수정**: U-Blox 계열 `TwistWithCovarianceStamped`는 ENU(x=East, y=North, z=Up)로 발행됨. 
  기존 구현은 x=North, y=East로 해석하여 남쪽 주행 시 `y<0`를 서쪽 주행으로 잘못 해석하는 문제가 있었음. 
  `gnssVelCallback()`에서 ENU→NED 매핑을 다음과 같이 수정하여 일관화함: `vN=y, vE=x, vD=-z`.
  - 코드 참조: `INS/gps_imu_fusion/src/ekf_fusion_node.cpp:483-485` (수정됨)
  - 증상 예: 남쪽 주행 시 `linear: {x≈0, y<0}` → ENU 기준 정상, 기존 코드에선 축 뒤바뀜으로 헤딩 오류 유발.

## 5. 남아있는 문제

### 5.1 High Priority
- **시간 동기화 부재**: IMU와 GPS 타임스탬프 동기화 없음
  - 상세 계획: [시간 동기화 구현 계획](./time_synchronization_plan.md) (예정)
- **Lever arm 보정**: ✅ 완료 (2025.08.27)
  - GPS position/velocity 및 IMU 가속도 lever arm 보정 구현 완료
  - TF 방향 문제 및 프레임 불일치 수정 완료
- **ZUPT 관련 문제**: 정지/이동 전환 시 불안정성 발생
  - IMU 정지 감지와 GPS 속도 기반 판단 간 충돌
  - 전환 상태(transition state) 로직 개선 필요
  - 히스테리시스 파라미터 재조정 필요
- **IMU calibration 미적용**: ~~JSON 파일 미사용~~ (해결됨)

### 5.2 Medium Priority
- **오류 복구 메커니즘 부재**: 센서 실패 시 대응 로직 없음
- **초기화 순서 문제**: GPS/GNSS velocity 모두 필요
- **map → odom TF 미발행**: SLAM 통합 시 필요

## 6. Lever Arm 보정 구현 완료 (2025.08.27)

### 6.1 구현된 보정 항목
- ✅ **GPS position lever arm**: `p_base = p_gps - R_wb * t_bg`
- ✅ **GPS velocity lever arm**: `v_base = v_gps - ω × (R_wb * t_bg)`  
- ✅ **IMU acceleration lever arm**: `a_base = a_imu - α×r - ω×(ω×r)`
- ✅ **TF 방향 수정**: `lookupTransform("base_link", source)` 형식으로 통일
- ✅ **프레임 일치 보장**: gnssVelCallback에서 base frame 각속도 사용

### 6.2 미구현 항목
- **고도 오프셋**: base_link 지면 높이(0.235m) 미적용
- **Higher-order terms**: 큰 lever arm(>0.5m)에 대한 고차항 미고려



## 7. 결론

### 7.1 현재 상태 요약
- **기본 기능 구현 완료**: EKF, ZUPT, GPS heading 융합
- **스레드 안전성 확보**: mutex 올바르게 구현
- **자동차 특화**: 평지 주행 가정, yaw rate 중심

### 7.2 주요 개선 필요사항
1. **ZUPT 안정성 개선**: 정지/이동 전환 시 불안정성 해결 필요
2. **시간 동기화**: 센서 간 타임스탬프 동기화
3. **고도 오프셋 추가**: 타이어 반지름 고려
4. **오류 복구**: 센서 실패 시 graceful degradation

### 7.3 평가
테스트 및 개발 목적으로 충분히 사용 가능한 상태. Lever arm 보정 완료로 정확도가 크게 향상됨. ZUPT 안정성 개선과 시간 동기화가 추가되면 프로덕션 수준 도달 가능.
