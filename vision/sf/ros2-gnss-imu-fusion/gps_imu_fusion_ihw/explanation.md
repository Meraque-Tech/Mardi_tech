
# GPS + IMU 센서 퓨전을 위한 Kalman Filter 정리
(코드 <-> 이론 매핑)

---

## 1. 상태 벡터(State Vector)

| 구분 | 수식 | 설명 | 구현 위치 |
|------|------|------|----------|
| **PDF 제안** | **x₆ = [x y ψ v<sub>x</sub> v<sub>y</sub> ω]ᵀ**   | 자이로 출력 ω(=yaw‑rate)까지 추정하여 **바이어스 보정**까지 고려 | – |
| **현재 코드** | **x₅ = [x y v<sub>x</sub> v<sub>y</sub> ψ]ᵀ** | ω는 **제어 입력**으로 직접 사용 → 행렬 차원 축소 | `kalman_filter.hpp : Vector5d x_` |

> **왜 5‑state를 택했는가?**  
> * 계산 가볍고 자이로 bias가 무시 가능할 때 실전에서 충분.  
> * 장·단기는 §6 참고.

---

## 2. 핵심 행렬 정의·역할·구현

| 기호 | 의미 / 공식 | 코드 구현 (주요 라인) | 주석 |
|------|-------------|----------------------|-------|
| **A** (상태 전이) | `A = I; A₀₂ = dt; A₁₃ = dt` (상수 속도 모델) | `predict(): A_(0,2)=dt; A_(1,3)=dt;` | 위치 ← 속도 적분 |
| **B** (제어 입력) | ψ(=yaw) 로테이션 반영 `B(dt,ψ)` | `computeBandQ()` | 가속·ω 적분 → x, y, v, ψ 갱신 |
| **Q** (프로세스 잡음) | `Q = B Σ Bᵀ`,  Σ = diag(σ<sub>a</sub>², σ<sub>a</sub>², σ<sub>ω</sub>²) | `computeBandQ()` | IMU 노이즈가 상태로 퍼짐 |
| **H** (측정) | `H = [[1,0,0,0,0],[0,1,0,0,0]]` | 생성자에서 할당 | GPS는 x,y만 측정 |
| **R** (측정 잡음) | `R = I·gps_cov` (RTK 기준 0.000196 m²) | `gpsCallback()` | GPS 신뢰도 |
| **P** (오차 공분산) | 초기 `diag(1,1,0.4,0.4,0.1)` | `gpsCallback()` 첫 프레임 | 위치↑ 속도·ψ↓ |

---

## 3. 알고리즘 단계 & 코드 흐름

### 3‑1 예측(Predict) – IMU 콜백
```cpp
/* 입력 */
Eigen::Vector2d acc_body(ax, ay);
double omega_z = msg->angular_velocity.z;

/* 예측 */
kf_->predict(acc_body, omega_z, dt);
```
1. **dt 방어**: 0 < dt ≤ 0.1 아닌 경우 0.01 고정.  
2. **A, B, Q 계산** → `x⁻ = A x + B u`, `P⁻ = A P Aᵀ + Q`.  
3. **ψ 각도 래핑** −π~π.

### 3‑2 갱신(Update) – GPS 콜백
```cpp
Eigen::Vector2d gps_xy(x, y);
kf_->update(gps_xy, R);
```
1. 혁신 `y = z – H x⁻`.  
2. `S = H P⁻ Hᵀ + R`, `K = P⁻ Hᵀ S⁻¹`.  
3. `x⁺ = x⁻ + K y`, `P⁺ = (I – K H) P⁻`.  
4. ψ 래핑.

---

## 4. ROS2 노드 배선

| 토픽 | 역할 | QoS | 연결 메서드 |
|------|------|-----|-------------|
| `/ouster/imu` | IMU 데이터(100 Hz) | `SensorDataQoS` | `imuCallback()` |
| `/local_xy` | GPS 위치 | Reliable(10) | `gpsCallback()` |
| `/odometry/fusion` | 필터 결과 | – | `publishOutputs()` |
| `/global_yaw` | ψ(rad) 단일 토픽 | – | `publishOutputs()` |
| TF | map → base | – | 조건부 |

---

## 5. 5‑state vs 6‑state 선택 가이드

| 항목 | 5‑state(현재) | 6‑state(PDF) |
|------|---------------|--------------|
| 모델 크기 | 작음(5×5) | 큼(6×6) |
| 자이로 bias 보정 | × (외부 처리 필요) | ○ (상태로 추정) |
| 계산 부하 | 낮음 | 높음 |
| 장기 ψ 드리프트 | 가능성 ↑ | 완화 |
| 현장 튜닝 난이도 | 간단 | 복잡 |

> **권장**  
> * **초기 개발·저비용 IMU** → 5‑state.  
> * **장기 항법·저속 주행** 또는 **bias 눈에 띄는 환경** → 6‑state + bias 항 확장.

---

## 6. 주요 파라미터 튜닝 팁

* **σ<sub>a</sub>, σ<sub>ω</sub>** : IMU 노이즈 효과. 로그에서 **Innovation** 분포 보고 조정.  
* **gps_cov** (*R*) : RTK → 1e‑4 수준, 일반 GNSS → 수 m² 수준까지 증가.  
* **P₀** : 초기 정지 가정이면 v<sub>x</sub>, v<sub>y</sub> 오차를 크게 (0.4~1.0).

---

## 7. 참고 식 모음

| 단계 | 식 |
|------|----|
| Predict | `x⁻ = A x + B u`  `P⁻ = A P Aᵀ + Q` |
| Update | `K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹`<br>`x⁺ = x⁻ + K (z – H x⁻)`<br>`P⁺ = (I – K H) P⁻` |

각 식과 행렬 기호는 위 표 §2 정의와 1:1 대응한다.

---

## 8. 결론

* **현 코드**는 PDF 이론을 거의 그대로 구현하되, **ω(자이로) bias가 크지 않다는 전제**로 5‑state를 선택.  
* 실전에서 yaw drift가 문제되면 **6‑state 확장**을 추천.  
* 나머지 행렬/공식은 PDF와 완전 일치하며 구현도 정확하다.

필터 구조·소스 매핑이 한눈에 보이도록 정리했으니, 튜닝·확장 시 참조하세요!