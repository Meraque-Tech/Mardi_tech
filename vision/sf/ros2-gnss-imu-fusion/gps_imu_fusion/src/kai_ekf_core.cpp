#include "gps_imu_fusion/kai_ekf_core.hpp"
#include <cmath>
#include "rclcpp/rclcpp.hpp"  
namespace kai {

KaiEkfCore::KaiEkfCore() {
  params_ = EkfParams();
  
  grav(2,0) = GRAVITY;
  
  H.block(0,0,5,5) = Eigen::Matrix<float,5,5>::Identity();
  H(6,8) = 1.0f;  // heading 측정은 상태 x(8) (yaw)와 연결
  
  updateProcessNoiseMatrix();
  
  resetCovarianceMatrix();

  RCLCPP_INFO(rclcpp::get_logger("KaiEkfCore"), "EKF Core 초기화 완료");
}

void KaiEkfCore::setParameters(const EkfParams& params) {
  params_ = params;
  
  updateProcessNoiseMatrix();
  resetCovarianceMatrix();

  RCLCPP_INFO(rclcpp::get_logger("KaiEkfCore"), "파라미터 업데이트 완료");
}

void KaiEkfCore::setStationary(bool stationary) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  stationary_mode_ = stationary;
}

void KaiEkfCore::setZuptNoiseScale(float scale) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  zupt_noise_scale_ = std::max(0.0f, scale);
}

void KaiEkfCore::updateProcessNoiseMatrix() {
  Rw.setZero();
  Rw.block(0,0,3,3) = powf(params_.accel_noise, 2.0f) * Eigen::Matrix<float,3,3>::Identity();
  Rw.block(3,3,3,3) = powf(params_.gyro_noise, 2.0f) * Eigen::Matrix<float,3,3>::Identity();
  Rw.block(6,6,3,3) = 2.0f * powf(params_.accel_bias_noise, 2.0f) / params_.accel_bias_tau * Eigen::Matrix<float,3,3>::Identity();
  Rw.block(9,9,3,3) = 2.0f * powf(params_.gyro_bias_noise, 2.0f) / params_.gyro_bias_tau * Eigen::Matrix<float,3,3>::Identity();
  
  R.setZero();
  R.block(0,0,2,2) = powf(params_.gps_pos_noise_ne, 2.0f) * Eigen::Matrix<float,2,2>::Identity();
  R(2,2) = powf(params_.gps_pos_noise_d, 2.0f);
  R.block(3,3,2,2) = powf(params_.gps_vel_noise_ne, 2.0f) * Eigen::Matrix<float,2,2>::Identity();
  R(5,5) = powf(params_.gps_vel_noise_d, 2.0f);
  R(6,6) = powf(gps_heading_noise_, 2.0f);  // heading 노이즈

  // 정지 모드에서는 yaw/gyro-bias-z에 해당하는 프로세스 노이즈를 추가로 축소
  if (stationary_mode_) {
    // Gs에서 yaw에 해당하는 컬럼은 (6~8) 회전 상태, 여기서는 Rw의 자이로 Z 노이즈와 bias Z 노이즈를 줄여 Q를 작게 만듦
    Rw(5,5) *= zupt_noise_scale_;   // gyro Z
    Rw(11,11) *= zupt_noise_scale_; // gyro bias Z
  }

  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "프로세스 노이즈 행렬 업데이트 완료");
}

void KaiEkfCore::resetCovarianceMatrix() {
  P.setZero();
  P.block(0,0,3,3) = powf(params_.init_pos_unc, 2.0f) * Eigen::Matrix<float,3,3>::Identity();
  P.block(3,3,3,3) = powf(params_.init_vel_unc, 2.0f) * Eigen::Matrix<float,3,3>::Identity();
  P.block(6,6,2,2) = powf(params_.init_att_unc, 2.0f) * Eigen::Matrix<float,2,2>::Identity();
  P(8,8) = powf(params_.init_hdg_unc, 2.0f);
  P.block(9,9,3,3) = powf(params_.init_accel_bias_unc, 2.0f) * Eigen::Matrix<float,3,3>::Identity();
  P.block(12,12,3,3) = powf(params_.init_gyro_bias_unc, 2.0f) * Eigen::Matrix<float,3,3>::Identity();

  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "공분산 행렬 초기화 완료");
}


void KaiEkfCore::ekfInit(uint64_t time, 
                    double vn, double ve, double vd, 
                    double lat, double lon, double alt,
                    float p, float q, float r, 
                    float ax, float ay, float az,
                    float hx, float hy, float hz) {
  // 자이로 바이어스 초기화 (0으로 시작)
  gbx = 0.0f;
  gby = 0.0f;
  gbz = 0.0f;
  
  std::tie(theta, phi, psi) = getPitchRollYaw(ax, ay, az, hx, hy, hz);
  
  // 자동차 평지 주행 가정: roll과 pitch를 0으로 고정
  phi = 0.0f;    // roll = 0
  theta = 0.0f;  // pitch = 0
  
  quat = eulerToQuaternion(psi, theta, phi);
  
  lat_ins = lat;
  lon_ins = lon;
  alt_ins = alt;
  vn_ins = vn;
  ve_ins = ve;
  vd_ins = vd;
  
  f_b(0,0) = ax;
  f_b(1,0) = ay;
  f_b(2,0) = az;
  
  _tprev = time;

  RCLCPP_INFO(rclcpp::get_logger("KaiEkfCore"), "EKF 초기화: time=%lu, lat=%.6f, lon=%.6f, alt=%.2f", time, lat, lon, alt);
}

void KaiEkfCore::ekfUpdate(uint64_t time,
                     double vn, double ve, double vd,
                     double lat, double lon, double alt,
                     float p, float q, float r,
                     float ax, float ay, float az,
                     float hx, float hy, float hz) {
  if (!initialized_) {
    ekfInit(time, vn, ve, vd, lat, lon, alt, p, q, r, ax, ay, az, hx, hy, hz);
    initialized_ = true;
  } else {
    // 시간이 역행하거나 동일한 경우 건너뜀
    if (time <= _tprev) {
      RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "중복된 시간 또는 시간 역행: time=%lu, _tprev=%lu", time, _tprev);
      return;
    }
    
    float dt = ((float)(time - _tprev)) / 1e6;
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "dt: %f 초", dt);
    
    updateImuData(ax, ay, az, p, q, r);
    
    updateIns();
    
    dq(0) = 1.0f;
    dq(1) = 0.0f;  // roll 변화율 = 0 (자동차 평지 주행)
    dq(2) = 0.0f;  // pitch 변화율 = 0 (자동차 평지 주행)
    dq(3) = 0.5f * om_ib(2,0) * dt;  // yaw 변화율만 사용
    quat = quatMultiply(quat, dq);
    quat.normalize();
    
    if (quat(0) < 0) {
      quat = -1.0f * quat;
    }
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "업데이트된 쿼터니언: [%.3f, %.3f, %.3f, %.3f]", quat(0), quat(1), quat(2), quat(3));
    
    C_N2B = quat2dcm(quat);
    C_B2N = C_N2B.transpose();
    
    std::tie(phi, theta, psi) = quaternionToEuler(quat);
    
    dx = C_B2N * f_b + grav;
    vn_ins += dt * dx(0,0);
    ve_ins += dt * dx(1,0);
    vd_ins += dt * dx(2,0);
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "업데이트된 INS 속도: vn=%.3f, ve=%.3f, vd=%.3f", vn_ins, ve_ins, vd_ins);
    
    Eigen::Matrix<double,3,1> V_temp;
    V_temp << vn_ins, ve_ins, vd_ins;
    dxd = llaRate(V_temp, lat_ins, alt_ins);
    lat_ins += dt * dxd(0,0);
    lon_ins += dt * dxd(1,0);
    alt_ins += dt * dxd(2,0);
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "업데이트된 INS 위치: lat=%.6f, lon=%.6f, alt=%.2f", lat_ins, lon_ins, alt_ins);
    
    updateJacobianMatrix();
    
    updateProcessNoiseAndCovariance(dt);
    
    lla_gps(0,0) = lat;
    lla_gps(1,0) = lon;
    lla_gps(2,0) = alt;
    
    V_gps(0,0) = vn;
    V_gps(1,0) = ve;
    V_gps(2,0) = vd;
    
    updateIns();
    
    updateMeasurementResidual();
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "측정 잔차 업데이트 완료");
    
    K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
    
    P = (Eigen::Matrix<float,15,15>::Identity() - K * H) * P * (Eigen::Matrix<float,15,15>::Identity() - K * H).transpose() + 
        K * R * K.transpose();
    
    x = K * y;
    
    update15StatesAfterKf();
    
    _tprev = time;
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "EKF 업데이트 완료, 새로운 시간: %lu", time);
  }
}

void KaiEkfCore::imuUpdateEkf(uint64_t time, const ImuData& imu) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  imuDat = imu;
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "IMU 데이터 업데이트 수신");
  
  ekfUpdate(time, gpsVel.vN, gpsVel.vE, gpsVel.vD,
           gpsCoor.lat, gpsCoor.lon, gpsCoor.alt,
           imuDat.gyroX, imuDat.gyroY, imuDat.gyroZ,
           imuDat.accX, imuDat.accY, imuDat.accZ,
           imuDat.hX, imuDat.hY, imuDat.hZ);
}

void KaiEkfCore::gpsCoordinateUpdateEkf(const GpsCoordinate& coor) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  gpsCoor = coor;
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "GPS 좌표 업데이트: lat=%.6f, lon=%.6f, alt=%.2f", coor.lat, coor.lon, coor.alt);
}

void KaiEkfCore::gpsVelocityUpdateEkf(const GpsVelocity& vel) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  gpsVel = vel;
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "GPS 속도 업데이트: vN=%.3f, vE=%.3f, vD=%.3f", vel.vN, vel.vE, vel.vD);
}

void KaiEkfCore::setGpsHeading(float heading, bool valid) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  gps_heading_ = heading;
  use_gps_heading_ = valid;
  
  if (valid) {
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "GPS heading 설정: %.2f deg", heading * 180.0 / M_PI);
  } else {
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "GPS heading 비활성화 (저속)");
  }
}

void KaiEkfCore::setGpsHeadingNoise(float stddev) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  // 표준편차를 직접 받아 R(6,6)에 분산으로 반영
  float clipped = std::max(0.0f, std::min(stddev, 1e5f));
  gps_heading_noise_ = clipped;
  R(6,6) = clipped * clipped;
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "GPS heading noise stddev 설정: %.6f (R66=%.6f)", clipped, R(6,6));
}

std::tuple<float,float,float> KaiEkfCore::getPitchRollYaw(float ax, float ay, float az, float hx, float hy, float hz) {
  // --- Pitch 계산 수정 ---
  float pitch_input = ax / GRAVITY;
  // 입력값이 -1.0 ~ 1.0 범위를 벗어나지 않도록 강제
  if (pitch_input > 1.0f) {
    pitch_input = 1.0f;
  } else if (pitch_input < -1.0f) {
    pitch_input = -1.0f;
  }
  float pitch = asinf(pitch_input);

  // --- Roll 계산 수정 ---
  float roll_input = -ay / (GRAVITY * cosf(pitch));
  // 입력값이 -1.0 ~ 1.0 범위를 벗어나지 않도록 강제
  if (roll_input > 1.0f) {
    roll_input = 1.0f;
  } else if (roll_input < -1.0f) {
    roll_input = -1.0f;
  }
  float roll = -asinf(roll_input);

  // ... (yaw 계산은 동일)
  float Bxc = hx * cosf(pitch) + (hy * sinf(roll) + hz * cosf(roll)) * sinf(pitch);
  float Byc = hy * cosf(roll) - hz * sinf(roll);
  float yaw = -atan2f(Byc, Bxc);
  
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "초기 자세 추정: pitch=%.3f, roll=%.3f, yaw=%.3f", pitch, roll, yaw);
  
  return std::make_tuple(pitch, roll, yaw);
}

void KaiEkfCore::updateIns() {
  lla_ins(0,0) = lat_ins;
  lla_ins(1,0) = lon_ins;
  lla_ins(2,0) = alt_ins;
  
  V_ins(0,0) = vn_ins;
  V_ins(1,0) = ve_ins;
  V_ins(2,0) = vd_ins;
}

void KaiEkfCore::updateMeasurementResidual() {
  pos_ecef_ins = lla2ecef(lla_ins);
  pos_ecef_gps = lla2ecef(lla_gps);
  
  pos_ned_gps = ecef2ned(pos_ecef_gps - pos_ecef_ins, lla_ins);
  
  y(0,0) = (float)(pos_ned_gps(0,0));
  y(1,0) = (float)(pos_ned_gps(1,0));
  y(2,0) = (float)(pos_ned_gps(2,0));
  y(3,0) = (float)(V_gps(0,0) - V_ins(0,0));
  y(4,0) = (float)(V_gps(1,0) - V_ins(1,0));
  y(5,0) = (float)(V_gps(2,0) - V_ins(2,0));
  
  // GPS heading 잔차 (유효한 경우에만)
  if (use_gps_heading_) {
    float heading_error = gps_heading_ - psi;
    // 각도 차이를 -π ~ π 범위로 정규화
    while (heading_error > M_PI) heading_error -= 2.0f * M_PI;
    while (heading_error < -M_PI) heading_error += 2.0f * M_PI;
    y(6,0) = heading_error;
    H(6,8) = 1.0f;  // heading 측정 활성화
  } else {
    y(6,0) = 0.0f;
    H(6,8) = 0.0f;  // heading 측정 비활성화
    R(6,6) = 1e10f;  // 매우 큰 노이즈로 무시
  }

  // 정지(ZUPT) 모드에서는 속도 0 제약(NE 방향) 유사 측정 추가 고려 가능 (단, 현재 구조는 y(0..6)에 맞춰 설계됨)
  // 간단히는 프로세스 측에서 억제하므로 측정은 추가하지 않음.
}

void KaiEkfCore::update15StatesAfterKf() {
  estmimated_ins = llaRate((x.block(0,0,3,1)).cast<double>(), lat_ins, alt_ins);
  lat_ins += estmimated_ins(0,0);
  lon_ins += estmimated_ins(1,0);
  alt_ins += estmimated_ins(2,0);
  
  vn_ins += x(3,0);
  ve_ins += x(4,0);
  vd_ins += x(5,0);
  
  dq(0,0) = 1.0f;
  dq(1,0) = 0.0f;  // roll 보정 = 0 (자동차 평지 주행)
  dq(2,0) = 0.0f;  // pitch 보정 = 0 (자동차 평지 주행)
  dq(3,0) = x(8,0);  // yaw 보정만 사용
  
  // 디버그: 업데이트 전 쿼터니언과 델타 확인
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "쿼터니언 업데이트 전: quat=[%.3f, %.3f, %.3f, %.3f], dq=[%.3f, %.3f, %.3f, %.3f]",
               quat(0), quat(1), quat(2), quat(3), dq(0), dq(1), dq(2), dq(3));
  
  quat = quatMultiply(quat, dq);
  
  // 정규화 전 확인
  float quat_norm_before = quat.norm();
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "정규화 전 쿼터니언 norm: %.6f", quat_norm_before);
  
  if (quat_norm_before > 0.001f) {  // 너무 작으면 정규화 불가
    quat.normalize();
  } else {
    RCLCPP_WARN(rclcpp::get_logger("KaiEkfCore"), "쿼터니언 norm이 너무 작음: %.6f, 리셋", quat_norm_before);
    quat << 1.0f, 0.0f, 0.0f, 0.0f;  // 단위 쿼터니언으로 리셋
  }
  
  // 정규화 후 확인
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "정규화 후 쿼터니언: [%.3f, %.3f, %.3f, %.3f]", 
               quat(0), quat(1), quat(2), quat(3));
  
  // NaN 체크
  if (std::isnan(quat(0)) || std::isnan(quat(1)) || std::isnan(quat(2)) || std::isnan(quat(3))) {
    RCLCPP_ERROR(rclcpp::get_logger("KaiEkfCore"), "쿼터니언에 NaN 검출! 리셋");
    quat << 1.0f, 0.0f, 0.0f, 0.0f;
  }
  
  std::tie(phi, theta, psi) = quaternionToEuler(quat);
  
  // 자동차 평지 주행 가정: roll과 pitch를 0으로 고정
  phi = 0.0f;    // roll = 0  
  theta = 0.0f;  // pitch = 0
  
  // 수정된 자세로 쿼터니언 재계산
  quat = eulerToQuaternion(psi, theta, phi);
  
  // 가속도계 바이어스는 EKF에서 추정하지 않음 (imu_preprocess에서 처리)
  // x(9,0), x(10,0), x(11,0)는 사용하지 않음
  
  gbx += x(12,0);
  gby += x(13,0);
  gbz += x(14,0);

  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "15 상태 변수 업데이트 완료");
}

void KaiEkfCore::updateImuData(float ax, float ay, float az, float p, float q, float r) {
  // IMU 전처리 노드에서 이미 바이어스가 제거되었으므로 그대로 사용
  f_b(0,0) = ax;
  f_b(1,0) = ay;
  f_b(2,0) = az;

  // 자이로 데이터에서 EKF가 추정한 바이어스 제거
  // 자동차 평지 주행: roll, pitch 각속도 = 0
  om_ib(0,0) = 0.0f;  // roll rate = 0
  om_ib(1,0) = 0.0f;  // pitch rate = 0
  // ZUPT: 정지 모드일 때 yaw rate를 0으로 강제하여 발산 억제
  if (stationary_mode_) {
    om_ib(2,0) = 0.0f;
  } else {
    om_ib(2,0) = r - gbz;  // yaw rate만 사용
  }

  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "IMU 데이터 업데이트: ax=%.3f, ay=%.3f, az=%.3f, p=%.3f, q=%.3f, r=%.3f", ax, ay, az, p, q, r);
}

void KaiEkfCore::updateProcessNoiseAndCovariance(float dt) {
  PHI = Eigen::Matrix<float,15,15>::Identity() + Fs * dt;
  
  Gs.setZero();
  Gs.block(3,0,3,3) = -C_B2N;  
  Gs.block(6,3,3,3) = -0.5f * Eigen::Matrix<float,3,3>::Identity();  
  Gs.block(9,6,6,6) = Eigen::Matrix<float,6,6>::Identity();  

  // 정지 모드에서는 yaw/gyro-bias-z 관련 프로세스 노이즈 축소를 반영
  Eigen::Matrix<float,12,12> Rw_effective = Rw;
  if (stationary_mode_) {
    Rw_effective(5,5) *= zupt_noise_scale_;   // gyro Z
    Rw_effective(11,11) *= zupt_noise_scale_; // gyro bias Z
  }

  Q = PHI * dt * Gs * Rw_effective * Gs.transpose();
  Q = 0.5f * (Q + Q.transpose());  

  P = PHI * P * PHI.transpose() + Q;
  P = 0.5f * (P + P.transpose());  

  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "프로세스 노이즈 및 공분산 시간 업데이트 완료");
}

void KaiEkfCore::updateJacobianMatrix() {
  Fs.setZero();
  
  Fs.block(0,3,3,3) = Eigen::Matrix<float,3,3>::Identity();
  
  Fs(5,2) = -2.0f * GRAVITY / EARTH_RADIUS;
  
  Fs.block(3,6,3,3) = -2.0f * C_B2N * skewSymmetric(f_b);
  
  Fs.block(3,9,3,3) = -C_B2N;
  
  Fs.block(6,6,3,3) = -skewSymmetric(om_ib);
  
  Fs.block(6,12,3,3) = -0.5f * Eigen::Matrix<float,3,3>::Identity();
  
  Fs.block(9,9,3,3) = -1.0f / params_.accel_bias_tau * Eigen::Matrix<float,3,3>::Identity();
  
  Fs.block(12,12,3,3) = -1.0f / params_.gyro_bias_tau * Eigen::Matrix<float,3,3>::Identity();

  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), "자코비안 행렬 업데이트 완료");
}

Eigen::Matrix<float,3,3> KaiEkfCore::skewSymmetric(Eigen::Matrix<float,3,1> v) {
  Eigen::Matrix<float,3,3> m;
  m(0,0) = 0.0f;     m(0,1) = -v(2,0);  m(0,2) = v(1,0);
  m(1,0) = v(2,0);   m(1,1) = 0.0f;     m(1,2) = -v(0,0);
  m(2,0) = -v(1,0);  m(2,1) = v(0,0);   m(2,2) = 0.0f;
  return m;
}

std::pair<double, double> KaiEkfCore::earthRadius(double lat) {
  double denom = fabs(1.0 - (ECCENTRICITY_SQ * pow(sin(lat), 2.0)));
  double Rew = EARTH_RADIUS / sqrt(denom); 
  double Rns = EARTH_RADIUS * (1.0 - ECCENTRICITY_SQ) / (denom * sqrt(denom));  
  return std::make_pair(Rew, Rns);
}

Eigen::Matrix<double,3,1> KaiEkfCore::llaRate(const Eigen::Matrix<double,3,1>& V, const Eigen::Matrix<double,3,1>& lla) {

  double Rew, Rns;
  Eigen::Matrix<double,3,1> lla_dot;
  
  std::tie(Rew, Rns) = earthRadius(lla(0,0));
  
  lla_dot(0,0) = V(0,0) / (Rns + lla(2,0));  
  lla_dot(1,0) = V(1,0) / ((Rew + lla(2,0)) * cos(lla(0,0)));  
  lla_dot(2,0) = -V(2,0);  
  
  return lla_dot;
}

Eigen::Matrix<double,3,1> KaiEkfCore::llaRate(const Eigen::Matrix<double,3,1>& V, double lat, double alt) {
  Eigen::Matrix<double,3,1> lla;
  lla(0,0) = lat;
  lla(1,0) = 0.0; 
  lla(2,0) = alt;
  return llaRate(V, lla);
}

Eigen::Matrix<double,3,1> KaiEkfCore::lla2ecef(const Eigen::Matrix<double,3,1>& lla) {
  double Rew;
  Eigen::Matrix<double,3,1> ecef;
  
  std::tie(Rew, std::ignore) = earthRadius(lla(0,0));
  
  ecef(0,0) = (Rew + lla(2,0)) * cos(lla(0,0)) * cos(lla(1,0));
  ecef(1,0) = (Rew + lla(2,0)) * cos(lla(0,0)) * sin(lla(1,0));
  ecef(2,0) = (Rew * (1.0 - ECCENTRICITY_SQ) + lla(2,0)) * sin(lla(0,0));
  
  return ecef;
}

Eigen::Matrix<double,3,1> KaiEkfCore::ecef2ned(const Eigen::Matrix<double,3,1>& ecef, const Eigen::Matrix<double,3,1>& pos_ref) {

  Eigen::Matrix<double,3,1> ned;
  
  ned(0,0) = -sin(pos_ref(0,0)) * cos(pos_ref(1,0)) * ecef(0,0) -
              sin(pos_ref(0,0)) * sin(pos_ref(1,0)) * ecef(1,0) +
              cos(pos_ref(0,0)) * ecef(2,0);
              
  ned(1,0) = -sin(pos_ref(1,0)) * ecef(0,0) +
               cos(pos_ref(1,0)) * ecef(1,0);
               
  ned(2,0) = -cos(pos_ref(0,0)) * cos(pos_ref(1,0)) * ecef(0,0) -
              cos(pos_ref(0,0)) * sin(pos_ref(1,0)) * ecef(1,0) -
              sin(pos_ref(0,0)) * ecef(2,0);
              
  return ned;
}

Eigen::Matrix<float,3,3> KaiEkfCore::quat2dcm(const Eigen::Matrix<float,4,1>& q) {
  Eigen::Matrix<float,3,3> C;
  
  float q0q0 = q(0,0) * q(0,0);
  float q0q1 = q(0,0) * q(1,0);
  float q0q2 = q(0,0) * q(2,0);
  float q0q3 = q(0,0) * q(3,0);
  float q1q1 = q(1,0) * q(1,0);
  float q1q2 = q(1,0) * q(2,0);
  float q1q3 = q(1,0) * q(3,0);
  float q2q2 = q(2,0) * q(2,0);
  float q2q3 = q(2,0) * q(3,0);
  float q3q3 = q(3,0) * q(3,0);
  
  C(0,0) = 2.0f * (q0q0 + q1q1) - 1.0f;
  C(0,1) = 2.0f * (q1q2 + q0q3);
  C(0,2) = 2.0f * (q1q3 - q0q2);
  
  C(1,0) = 2.0f * (q1q2 - q0q3);
  C(1,1) = 2.0f * (q0q0 + q2q2) - 1.0f;
  C(1,2) = 2.0f * (q2q3 + q0q1);
  
  C(2,0) = 2.0f * (q1q3 + q0q2);
  C(2,1) = 2.0f * (q2q3 - q0q1);
  C(2,2) = 2.0f * (q0q0 + q3q3) - 1.0f;
  
  return C;
}

Eigen::Matrix<float,4,1> KaiEkfCore::quatMultiply(const Eigen::Matrix<float,4,1>& p, const Eigen::Matrix<float,4,1>& q) {
  Eigen::Matrix<float,4,1> r;
  
  r(0,0) = p(0,0) * q(0,0) - p(1,0) * q(1,0) - p(2,0) * q(2,0) - p(3,0) * q(3,0);
  r(1,0) = p(0,0) * q(1,0) + p(1,0) * q(0,0) + p(2,0) * q(3,0) - p(3,0) * q(2,0);
  r(2,0) = p(0,0) * q(2,0) - p(1,0) * q(3,0) + p(2,0) * q(0,0) + p(3,0) * q(1,0);
  r(3,0) = p(0,0) * q(3,0) + p(1,0) * q(2,0) - p(2,0) * q(1,0) + p(3,0) * q(0,0);
  
  return r;
}

float KaiEkfCore::constrainAngle180(float angle) const {
  if (angle > M_PI) angle -= (2.0f * M_PI);
  if (angle < -M_PI) angle += (2.0f * M_PI);
  return angle;
}

float KaiEkfCore::constrainAngle360(float angle) const {
  angle = fmod(angle, 2.0f * M_PI);
  if (angle < 0) angle += 2.0f * M_PI;
  return angle;
}

Eigen::Matrix<float,4,1> KaiEkfCore::eulerToQuaternion(float yaw, float pitch, float roll) {
  float cy = cosf(yaw * 0.5f);
  float sy = sinf(yaw * 0.5f);
  float cp = cosf(pitch * 0.5f);
  float sp = sinf(pitch * 0.5f);
  float cr = cosf(roll * 0.5f);
  float sr = sinf(roll * 0.5f);
  
  Eigen::Matrix<float,4,1> q;
  q(0) = cr * cp * cy + sr * sp * sy;  
  q(1) = sr * cp * cy - cr * sp * sy;  
  q(2) = cr * sp * cy + sr * cp * sy;  
  q(3) = cr * cp * sy - sr * sp * cy;  
  
  return q;
}

std::tuple<float,float,float> KaiEkfCore::quaternionToEuler(const Eigen::Matrix<float,4,1>& q) {
  // -------------------- 이 함수를 수정합니다 --------------------
  float roll, pitch, yaw;
  
  float sinr_cosp = 2.0f * (q(0,0) * q(1,0) + q(2,0) * q(3,0));
  float cosr_cosp = 1.0f - 2.0f * (q(1,0) * q(1,0) + q(2,0) * q(2,0));
  roll = atan2f(sinr_cosp, cosr_cosp);
  
  // --- Pitch 계산 수정 ---
  float sinp = 2.0f * (q(0,0) * q(2,0) - q(3,0) * q(1,0));
  // 입력값이 -1.0 ~ 1.0 범위를 벗어나지 않도록 강제합니다.
  if (sinp > 1.0f) {
      sinp = 1.0f;
  } else if (sinp < -1.0f) {
      sinp = -1.0f;
  }
  pitch = asinf(sinp);
  // --- 여기까지 수정 ---
    
  float siny_cosp = 2.0f * (q(0,0) * q(3,0) + q(1,0) * q(2,0));
  float cosy_cosp = 1.0f - 2.0f * (q(2,0) * q(2,0) + q(3,0) * q(3,0));
  yaw = atan2f(siny_cosp, cosy_cosp);
  
  return std::make_tuple(roll, pitch, yaw);
}

void KaiEkfCore::setGpsCovariance(const std::array<double, 9>& pos_cov, const std::array<double, 9>& vel_cov) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  
  // 기존 R 행렬 값을 유지하고, 유효한 공분산만 업데이트
  // GPS 위치 공분산을 R 행렬에 반영 (NED 좌표계로 변환 필요)
  // ROS는 ENU, EKF는 NED 사용
  // E->N, N->E, U->-D 변환
  if (pos_cov[4] > 0 && pos_cov[0] > 0 && pos_cov[8] > 0) {
    R(0,0) = pos_cov[4];  // North (ROS의 North)
    R(1,1) = pos_cov[0];  // East (ROS의 East)  
    R(2,2) = pos_cov[8];  // Down (ROS의 Up을 반전)
  }
  
  // GPS 속도 공분산을 R 행렬에 반영
  if (vel_cov[4] > 0 && vel_cov[0] > 0 && vel_cov[8] > 0) {
    R(3,3) = vel_cov[4];  // vN
    R(4,4) = vel_cov[0];  // vE
    R(5,5) = vel_cov[8];  // vD
  }
  
  RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), 
    "GPS 공분산 업데이트: pos_ne=%.4f/%.4f, pos_d=%.4f, vel_ne=%.4f/%.4f, vel_d=%.4f",
    R(0,0), R(1,1), R(2,2), R(3,3), R(4,4), R(5,5));
}

void KaiEkfCore::setImuCovariance(const std::array<double, 9>& gyro_cov, const std::array<double, 9>& accel_cov) {
  std::unique_lock<std::shared_mutex> lock(shMutex);
  
  // IMU 공분산을 Rw(프로세스 노이즈)에 반영
  bool updated = false;
  
  // 가속도계 공분산
  if (accel_cov[0] > 0) {
    Rw(0,0) = accel_cov[0];  // accel X
    Rw(1,1) = accel_cov[4];  // accel Y
    Rw(2,2) = accel_cov[8];  // accel Z
    updated = true;
  }
  
  // 자이로 공분산
  if (gyro_cov[0] > 0) {
    Rw(3,3) = gyro_cov[0];  // gyro X
    Rw(4,4) = gyro_cov[4];  // gyro Y
    Rw(5,5) = gyro_cov[8];  // gyro Z
    updated = true;
  }
  
  if (updated) {
    RCLCPP_DEBUG(rclcpp::get_logger("KaiEkfCore"), 
      "IMU 공분산 업데이트: accel=%.6f/%.6f/%.6f, gyro=%.6f/%.6f/%.6f",
      Rw(0,0), Rw(1,1), Rw(2,2), Rw(3,3), Rw(4,4), Rw(5,5));
  }
}

std::array<double, 9> KaiEkfCore::getPositionCovariance() const {
  std::shared_lock<std::shared_mutex> lock(shMutex);
  std::array<double, 9> cov = {0};
  
  // EKF가 초기화되지 않았으면 기본값 반환
  if (!initialized_) {
    // 초기 불확실성 사용
    double init_var = params_.init_pos_unc * params_.init_pos_unc;
    cov[0] = init_var;
    cov[4] = init_var;
    cov[8] = init_var;
    return cov;
  }
  
  // P 행렬의 위치 부분 (0:2, 0:2) 추출
  // NED에서 ENU로 변환
  cov[0] = P(1,1);  // East variance (NED의 E)
  cov[4] = P(0,0);  // North variance (NED의 N)
  cov[8] = P(2,2);  // Up variance (NED의 -D)
  
  // 공분산 (off-diagonal)
  cov[1] = P(1,0);  // E-N covariance
  cov[3] = P(0,1);  // N-E covariance
  
  return cov;
}

std::array<double, 9> KaiEkfCore::getVelocityCovariance() const {
  std::shared_lock<std::shared_mutex> lock(shMutex);
  std::array<double, 9> cov = {0};
  
  // EKF가 초기화되지 않았으면 기본값 반환
  if (!initialized_) {
    double init_var = params_.init_vel_unc * params_.init_vel_unc;
    cov[0] = init_var;
    cov[4] = init_var;
    cov[8] = init_var;
    return cov;
  }
  
  // P 행렬의 속도 부분 (3:5, 3:5) 추출
  // NED에서 ENU로 변환
  cov[0] = P(4,4);  // vE variance
  cov[4] = P(3,3);  // vN variance
  cov[8] = P(5,5);  // vU variance (-vD)
  
  // 공분산
  cov[1] = P(4,3);  // vE-vN covariance
  cov[3] = P(3,4);  // vN-vE covariance
  
  return cov;
}

std::array<double, 9> KaiEkfCore::getOrientationCovariance() const {
  std::shared_lock<std::shared_mutex> lock(shMutex);
  std::array<double, 9> cov = {0};
  
  // EKF가 초기화되지 않았으면 기본값 반환
  if (!initialized_) {
    cov[0] = params_.init_att_unc * params_.init_att_unc;
    cov[4] = params_.init_att_unc * params_.init_att_unc;
    cov[8] = params_.init_hdg_unc * params_.init_hdg_unc;
    return cov;
  }
  
  // P 행렬의 자세 부분 (6:8, 6:8) 추출
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      cov[i*3 + j] = P(6+i, 6+j);
    }
  }
  
  return cov;
}

}