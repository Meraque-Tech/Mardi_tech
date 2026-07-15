#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

namespace gps_imu_fusion
{
  using Vector5d  = Eigen::Matrix<double,5,1>;
  using Matrix5d  = Eigen::Matrix<double,5,5>;
  using Matrix5x3 = Eigen::Matrix<double,5,3>;
  using Vector2d  = Eigen::Vector2d;
  using Matrix2d  = Eigen::Matrix2d;
  using Matrix2x5 = Eigen::Matrix<double,2,5>;

  class KalmanFilter
  {
  public:
    KalmanFilter(double dt_default=0.01,
                 double sigma_a=0.1,
                 double sigma_omega=0.0245);

    void setInitialState(const Vector5d& x0, const Matrix5d& P0);
    void predict(const Vector2d& acc_body, double omega_z, double dt);
    void update (const Vector2d& gps_xy,  const Matrix2d& R);
    const Vector5d& state() const { return x_; }

  private:
    void computeBandQ(double psi, double dt);

    Vector5d  x_;
    Matrix5d  P_, A_, Q_;
    Matrix5x3 B_;
    Matrix2x5 H_;
    double    sigma_a2_, sigma_omega2_, dt_default_;
  };
}
