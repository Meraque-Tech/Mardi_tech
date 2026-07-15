#include <gps_imu_fusion_ihw/kalman_filter.hpp>
#include <Eigen/Dense>
#include <cmath>

namespace gps_imu_fusion
{
using Eigen::Matrix3d;
using Eigen::MatrixXd;

/* ───────── 헬퍼: 각도 ‑π~π 래핑 ───────── */
static inline double wrapAngle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

/* -------------------------------------------------------------------------- */
KalmanFilter::KalmanFilter(double dt_default,
                           double sigma_a,
                           double sigma_omega)
: sigma_a2_(sigma_a * sigma_a),
  sigma_omega2_(sigma_omega * sigma_omega),
  dt_default_(dt_default)
{
  x_.setZero();
  P_.setIdentity();
  P_ *= 1e3;

  A_.setIdentity();
  H_.setZero();
  H_(0,0) = 1.0;
  H_(1,1) = 1.0;

  B_.setZero();
  Q_.setZero();
}

/* -------------------------------------------------------------------------- */
void KalmanFilter::setInitialState(const Vector5d &x0, const Matrix5d &P0)
{
  x_ = x0;
  P_ = P0;
}

/* -------------------------------------------------------------------------- */
void KalmanFilter::computeBandQ(double psi, double dt)
{
  const double c = std::cos(psi);
  const double s = std::sin(psi);
  const double dt2 = dt*dt;
  const double half_dt2 = 0.5*dt2;

  B_.setZero();
  B_(0,0) =  half_dt2 * c;  B_(0,1) = -half_dt2 * s;
  B_(1,0) =  half_dt2 * s;  B_(1,1) =  half_dt2 * c;
  B_(2,0) =       dt * c;   B_(2,1) =      -dt * s;
  B_(3,0) =       dt * s;   B_(3,1) =       dt * c;
  B_(4,2) = dt;

  Matrix3d Sigma = Matrix3d::Zero();
  Sigma(0,0) = Sigma(1,1) = sigma_a2_;
  Sigma(2,2) = sigma_omega2_;

  Q_ = B_ * Sigma * B_.transpose();
}

/* -------------------------------------------------------------------------- */
void KalmanFilter::predict(const Vector2d &acc_body,
                           double omega_z,
                           double dt)
{
  if (dt <= 0.0 || std::isnan(dt))
    dt = dt_default_;

  A_(0,2) = dt;
  A_(1,3) = dt;

  computeBandQ(x_(4), dt);

  Eigen::Vector3d u;
  u << acc_body(0), acc_body(1), omega_z;

  x_ = A_ * x_ + B_ * u;
  x_(4) = wrapAngle(x_(4));               /* ★ ψ 래핑 */

  P_ = A_ * P_ * A_.transpose() + Q_;
}

/* -------------------------------------------------------------------------- */
void KalmanFilter::update(const Vector2d &gps_xy, const Matrix2d &R)
{
  Vector2d y = gps_xy - H_ * x_;
  Matrix2d S = H_ * P_ * H_.transpose() + R;

  Eigen::Matrix<double,5,2> K = P_ * H_.transpose() * S.inverse();

  x_ += K * y;
  x_(4) = wrapAngle(x_(4));               /* ★ ψ 래핑 */

  Matrix5d I = Matrix5d::Identity();
  P_ = (I - K * H_) * P_;
}

} // namespace gps_imu_fusion
