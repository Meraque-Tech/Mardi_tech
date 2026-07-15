# Global Yaw Estimation Nodes

이 문서는 `gps_imu_fusion_ihw` 패키지에 포함된 **global yaw** 추정 노드 두 가지의 알고리즘을 비교‧정리한 것입니다.

---

## 1. `global_yaw_integrator_node.cpp`

**구독 토픽**  
- `/local_xy` (`geometry_msgs/PointStamped`)  
- `/imu/processed` (`sensor_msgs/Imu`)  

**퍼블리시**  
- `/global_yaw` (`std_msgs/Float32`, 단위: rad)

**주요 파라미터**  
- `v_min` [m/s]: GPS 헤딩을 신뢰할 최소 속도  
- `dist_min` [m]: 초기화용 최소 이동 거리  
- `debug` [bool]

**알고리즘 흐름**  
1. **초기 yaw (ψ₀) 결정**  
   - 누적 이동 거리 ≥ `dist_min` 이고 속도 ≥ `v_min`일 때 단 한 번만  
   -  
     $$
     \psi_0 = \mathrm{atan2}(\Delta y, \Delta x)
     $$

2. **IMU 적분 단계**  
   - 각 콜백에서 아래와 같이 업데이트  
     $$
     \psi \leftarrow \mathrm{wrap}(\psi + \omega_z \cdot \Delta t)
     $$

3. `wrap()` 함수는 –π ~ π 범위로 보정

**장점**  
- 구조가 단순하고 계산 비용이 매우 적음

**한계**  
- 초기화 이후 **IMU 드리프트**가 누적되어 장기 주행 시 yaw 오차가 커짐

---

## 2. `global_yaw_complementary_node.cpp`

**구독 토픽**  
- `/local_xy`, `/imu/processed` (동일)

**퍼블리시**  
- `/global_yaw` (동일)

**주요 파라미터**  
- `v_min`, `dist_min` (동일)  
- `k_corr` [0~1]: GPS 보정 게인  
- `gps_timeout` [s]: GPS 최신성 한계  
- `debug`

**알고리즘 흐름**  
1. **초기 yaw (ψ₀) 결정**  
   - integrator와 동일  
   -  
     $$
     \psi_0 = \mathrm{atan2}(\Delta y, \Delta x)
     $$

2. **IMU 적분**  
   $$
   \psi \leftarrow \mathrm{wrap}(\psi + \omega_z \cdot \Delta t)
   $$

3. **GPS 보정 (Complementary Filter)**  
   조건:
   - 최근 GPS yaw가 `gps_timeout` 안에 있고  
   - 속도 ≥ `v_min`이면  

   오차 계산 및 보정:
   $$
   \delta = \mathrm{wrap}(\psi_{\mathrm{GPS}} - \psi)
   $$
   $$
   \psi \leftarrow \mathrm{wrap}(\psi + k_{\mathrm{corr}} \cdot \delta)
   $$

**장점**  
- 저주파(GPS) + 고주파(IMU) 정보 결합 → **장기 드리프트 억제**

**한계**  
- `k_corr`, `gps_timeout` 튜닝 필요  
- GPS 노이즈가 많을 경우 보정 품질 저하 가능

---

## 3. wrap 함수 정의

```cpp
inline double wrap(double a)
{
  return std::atan2(std::sin(a), std::cos(a)); // –π ≤ a < π
}
```

---

## 4. 비교 요약

| 구분 | Integrator | Complementary |
|------|------------|---------------|
| 초기 yaw | GPS Δx,Δy 이용 (동일) | GPS Δx,Δy 이용 (동일) |
| 업데이트 | IMU ω_z 적분만 | IMU 적분 + GPS 오차 보정 |
| 파라미터 | 2개 (`v_min`, `dist_min`) | 4개 (`v_min`, `dist_min`, `k_corr`, `gps_timeout`) |
| 장기 안정성 | 낮음 (드리프트 누적) | 높음 (드리프트 억제) |
| 구현 난이도 | 매우 쉬움 | 보통 (보정 로직 추가) |

---

## 5. 튜닝 가이드

| 파라미터 | 권장 범위 | 영향 |
|----------|----------|-------|
| `k_corr` | 0.01 – 0.05 | 증가 → GPS 보정 빠름 (노이즈 영향↑) |
| `gps_timeout` | GPS 주기 × 1.5 | 타임아웃 짧을수록 오랫된 GPS yaw 무시 |
| `v_min` | 실제 감속 한계 이하 | 너무 크면 정속·저속 시 yaw 보정 안 됨 |
| `dist_min` | 1 m ± | 주행 시작 직후 잘못된 초기 yaw 방지 |

---

> **Tip**  Complementary Filter로 드리프트를 충분히 억제할 수 없는 경우, IMU 바이어스까지 추정하는 EKF/UKF로 확장하는 것을 고려하세요.