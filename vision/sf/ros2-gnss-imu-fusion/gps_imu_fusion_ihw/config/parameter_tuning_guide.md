# gps_imu_fusion.yaml — Parameter Guide

이 파일은 `sensor_fusion_node`가 사용하는 모든 ROS 2 파라미터와
그 의미·권장 범위를 상세히 설명합니다.  
실험/현장 환경에 맞추어 값을 조정하면서 EKF의 응답성을 최적화할 수 있습니다.

| Key | 기본값 | 단위 | 역할 (행렬) | 값 ↑ 영향 | 값 ↓ 영향 |
|-----|--------|------|-------------|-----------|-----------|
| **sigma_a** | `0.25` | m·s⁻² (1 σ) | 과정 잡음 `Q`의 **선형 가속** 항 | GPS 의존↑, 위치 노이즈↓ <br> 반응 지연↑ | IMU 예측 의존↑, 빠른 반응 <br> 진동↑ |
| **sigma_omega** | `0.03` | rad·s⁻¹ (1 σ) | `Q`의 **요속**(ω_z) 항 | 헤딩을 GPS가 자주 보정 <br> 드리프트↓, 지연↑ | IMU 적분 의존↑, 빠름 <br> 드리프트 위험 |
| **gps_cov** | `0.000196` | m² | 관측 잡음 `R` (x,y) | GPS를 **덜** 믿음 <br> 터널·도심에서 안전 | GPS를 **더** 믿음 <br> RTK 등 정밀 환경 |
| **dt_default** | `0.01` | s | IMU 타임스탬프 이상 시 사용 | – | – |
| **map_frame** | `"reference"` | – | TF parent frame | – | – |
| **base_frame** | `"gps_antenna"` | – | TF child frame | – | – |
| **publish_tf** | `true` | bool | TF 브로드캐스트 on/off | TF 시각화 가능 | TF 충돌 방지 |

---

## 1.  튜닝 절차

1. **기본값**으로 주행 로그를 녹화한다.
2. `rqt_plot` 으로  
   * `/global_yaw` (EKF)  
   * GPS‑only yaw  
   * 위치 오차(ground truth 가 있으면)  
   를 동시에 관찰.
3. 다음 상황별로 파라미터를 조정한다.

### 1.1 헤딩(ψ) 드리프트가 크다
* `sigma_omega` **↑** (e.g. 0.04 → 0.06)  
  → IMU 요속을 “잡음이 크다”고 가정 → GPS 보정 빈도 ↑

### 1.2 헤딩 반응이 느리다 (코너에서 따라오지 못함)
* `sigma_omega` **↓** (0.03 → 0.02)  
  또는 `gps_cov` **↑**  
  → IMU 적분을 더 신뢰 (빠르지만 소음↑)

### 1.3 위치가 지그재그/진동
* `sigma_a` **↑** (0.25 → 0.35)  
  → 선형 가속 잡음 증가 → 속도 스무딩

### 1.4 코너 안쪽으로 파고듦 (둔함)
* `sigma_a` **↓** (0.25 → 0.15)

### 1.5 GPS 점프 (터널·수목 등)
* `gps_cov` **↑** (0.000196 → 0.002)  
  → GPS 측정을 “의심”하여 IMU 예측 유지

### 1.6 RTK cm급 환경
* `gps_cov` **↓** (0.00005)  
  → 매우 정확한 GPS를 적극 반영

---

## 2.  실시간 파라미터 변경

```bash
# 현재 값 확인
ros2 param get /gps_imu_sensor_fusion sigma_omega

# 값 변경
ros2 param set /gps_imu_sensor_fusion sigma_omega 0.045
```

변경된 값은 노드 재시작 시 사라지므로,  
안정값을 찾으면 **yaml** 파일에 기록 후 재배포.

---
