#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
	
  // State dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = 7;

	// Spreading parameter
	lambda_ = 3 - n_aug_;

	// Matrix to hold sigma points
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Weights
	weights_ = VectorXd(2*n_aug_+1);

  weights_(0) = lambda_/(lambda_+n_aug_);
  double std_weight = 0.5/(lambda_+n_aug_);

  for (int i = 1; i < 2*n_aug_+1; ++i) 
  {  
    weights_(i) = std_weight;
  }

	// Timestamp
	time_us_ = 0;

  is_initialized_ = false;

}

UKF::~UKF() {}


void UKF::initEstimateLaser(MeasurementPackage measPkg)
{
  std::cout << "initEstimateLaser ..." << std::endl;

  float x = measPkg.raw_measurements_(0);
  float y = measPkg.raw_measurements_(1);
  x_ << x, y, 0, 0, 0;

  P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
        0, std_laspy_*std_laspy_, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
}

void UKF::initEstimateRadar(MeasurementPackage measPkg)
{
  std::cout << "initEstimateRadar ..." << std::endl;

  float rho = measPkg.raw_measurements_(0);
  float phi = measPkg.raw_measurements_(1);
  float rho_dot = measPkg.raw_measurements_(2);

  float x = rho*cos(phi);
  float y = rho*sin(phi);
  double vx = rho_dot * cos(phi);
  double vy = rho_dot * sin(phi);
  double v = sqrt(vx * vx + vy * vy);

  x_ << x, y, v, 0, 0;

  P_ <<  std_radr_ * std_radr_, 0, 0, 0, 0,
         0, std_radr_ * std_radr_, 0, 0, 0,
         0, 0, std_radrd_ * std_radrd_, 0, 0,
         0, 0, 0, std_radrd_ * std_radrd_, 0,
         0, 0, 0, 0, std_radphi_*std_radphi_;  

}



void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  std::cout << "UKF::ProcessMeasurement " << std::endl;

  if (is_initialized_)
  {
    float dt = (meas_package.timestamp_ - time_us_)/1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    switch (meas_package.sensor_type_)
    {
      case MeasurementPackage::LASER:
                    UpdateLidar(meas_package);
                    break;
      case MeasurementPackage::RADAR:
                    UpdateRadar(meas_package);
                    break;
      default:
        std::cout << "ProcessMeasurement::processing: Invalid sensor type";
    }

  }
  else
  {
    switch (meas_package.sensor_type_)
    {
      case MeasurementPackage::LASER:
                    initEstimateLaser(meas_package);
                    break;
      case MeasurementPackage::RADAR:
                    initEstimateRadar(meas_package);
                    break;
      default:
        std::cout << "ProcessMeasurement::init: Invalid sensor type";
    }  
    is_initialized_ = true;
    std::cout << "initialized sensor " << meas_package.sensor_type_ << std::endl;
    time_us_ = meas_package.timestamp_;
  }

}

void UKF::Prediction(double delta_t) 
{
  std::cout << "UKF::Prediction... " << std::endl;

  VectorXd x_aug = VectorXd(n_aug_);                     // augmented state (mean) vector
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);             // augmented state covariance
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);  // sigma point matrix
  MatrixXd L = MatrixXd(n_aug_, n_aug_);
  
  x_aug = augmentState(x_, n_aug_);
  P_aug = augmentCovMatrix(P_, std_a_, std_yawdd_, n_aug_);

  //std::cout << x_aug << P_aug; 

  L = P_aug.llt().matrixL();
  Xsig_aug = augmentSigmaPoints(x_aug, lambda_, L, n_aug_);

  Xsig_pred_ = predictSigmaPoints(Xsig_aug, n_aug_, delta_t);
  //std::cout << Xsig_aug << Xsig_pred_ << std::endl;


  x_ = predictStateMean(Xsig_pred_);
  //std::cout <<std::endl << "predicted state x = " << std::endl << x_ << std::endl << " - - - - - - " << std::endl;
  P_ = predictCovMatrix(x_);
  std::cout << "predicted cov matrix = " << std::endl << P_ << std::endl << " - - - - - - " << std::endl;
}




void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  std::cout << "UKF::UpdateLidar ... " << std::endl;
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for sigma points in measurement space
  MatrixXd Z_sig = MatrixXd(z.size(), 2 * n_aug_ + 1);

  // transform sigma points into measurement space for lidar
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    // apply measurement model for lidar
    Z_sig(0,i) = p_x;                       
    Z_sig(1,i) = p_y;                                
  }

  VectorXd z_pred = VectorXd(z.size());
  z_pred = computeMeanPredictedMeasurement(z_pred, Z_sig);

  MatrixXd S = MatrixXd(z.size(),z.size());
  S = computeMeasurementCovMatrix(S, Z_sig, z_pred);

  // add measurement noise covariance matrix (noise is sensor specific)
  MatrixXd R = MatrixXd(z.size(),z.size());
  R <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  S = S + R;

  // create matrix for cross correlation Tc 
  MatrixXd Tc = MatrixXd(n_x_, z.size());
  Tc = computeCrosscorrelationMatrix(Tc, Z_sig, z_pred);

  updateState(Tc, S, z, z_pred);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  std::cout << "UKF::UpdateRadar ... " << std::endl;
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for sigma points in measurement space
  MatrixXd Z_sig = MatrixXd(z.size(), 2 * n_aug_ + 1);

  // transform sigma points for radar
  const int rho_pos = 0; const int phi_pos = 1; const int r_dot_pos = 2; const int yaw_pos = 3;
  for (int sigma_point = 0; sigma_point < 2 * n_aug_ + 1; ++sigma_point) 
  {  
    double p_x = Xsig_pred_(rho_pos,   sigma_point);
    double p_y = Xsig_pred_(phi_pos,   sigma_point);
    double v   = Xsig_pred_(r_dot_pos, sigma_point);
    double yaw = Xsig_pred_(yaw_pos,   sigma_point);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // apply measurement model for radar
    Z_sig(rho_pos,  sigma_point) = sqrt(p_x*p_x + p_y*p_y);    
    Z_sig(phi_pos,  sigma_point) = atan2(p_y,p_x);  

    if (sqrt(p_x*p_x + p_y*p_y) < 0.00001)
      Z_sig(r_dot_pos,sigma_point) = (p_x*v1 + p_y*v2) / 0.00001;
    else 
      Z_sig(r_dot_pos,sigma_point) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   
  }

  VectorXd z_pred = VectorXd(z.size());
  z_pred = computeMeanPredictedMeasurement(z_pred, Z_sig);

  MatrixXd S = MatrixXd(z.size(),z.size());
  S = computeMeasurementCovMatrix(S, Z_sig, z_pred);

  // add measurement noise covariance matrix (noise is sensor specific)
  MatrixXd R = MatrixXd(z.size(),z.size());
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, z.size());
  Tc = computeCrosscorrelationMatrix(Tc, Z_sig, z_pred);

  updateState(Tc, S, z, z_pred);
}




/* ------------------------------------------------------------------------------- */



  VectorXd UKF::augmentState(VectorXd &state, int dim)
  {
    VectorXd augState(dim);

    augState.head(5) = state;
    augState(5) = 0;
    augState(6) = 0;

    return augState;
  }

  MatrixXd UKF::augmentCovMatrix(MatrixXd &covMatrix, double stdDevAccelNoise, double stdDevYawAccelNoise, int dim)
  {
    MatrixXd augCovMatrix = MatrixXd(dim, dim);;

    augCovMatrix.fill(0.0);
    augCovMatrix.topLeftCorner(5,5) = covMatrix;
    augCovMatrix(5,5) = stdDevAccelNoise * stdDevAccelNoise;
    augCovMatrix(6,6) = stdDevYawAccelNoise * stdDevYawAccelNoise;
    return augCovMatrix;
  }

  MatrixXd UKF::augmentSigmaPoints(VectorXd &augState, double lambda, MatrixXd &L, int dim)
  {
    MatrixXd Xsig_aug = MatrixXd(dim, 2 * dim + 1);
   
    Xsig_aug.col(0)  = augState;
    for (int i = 0; i< dim; ++i) 
    {
      Xsig_aug.col(i+1)        = augState + sqrt(lambda + dim) * L.col(i);
      Xsig_aug.col(i+1+dim)    = augState - sqrt(lambda + dim) * L.col(i);
    }
    return Xsig_aug;
  }

 MatrixXd UKF::predictSigmaPoints(MatrixXd &Xsig_aug, int n_aug_, double dt)
  {
    for (int i = 0; i < 2*n_aug_+1; ++i) 
    {
      double p_x = Xsig_aug(0,i);
      double p_y = Xsig_aug(1,i);
      double v = Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yawd = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_yawdd = Xsig_aug(6,i);

      // predicted state values
      double px_p, py_p;

      bool predictCurvedMovement = fabs(yawd) > 0.001;

      if (predictCurvedMovement) {
          px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw) );
          py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
      } 
      else // we're headed on straight line
      {
          px_p = p_x + v*dt*cos(yaw);
          py_p = p_y + v*dt*sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd*dt;
      double yawd_p = yawd;

      // add noise
      px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
      py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
      v_p = v_p + nu_a*dt;

      yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
      yawd_p = yawd_p + nu_yawdd*dt;

      // write predicted sigma point into right column
      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yawd_p;
    }

    return Xsig_pred_;

  }

  VectorXd UKF::predictStateMean(MatrixXd &Xsig_pred_)
  {
    VectorXd state = VectorXd(n_x_);
    
    state.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)   
    {  
      state = state + weights_(i) * Xsig_pred_.col(i);
    }
    return state;
  }

  MatrixXd UKF::predictCovMatrix(const VectorXd& state)
  {
    MatrixXd P_pred = MatrixXd(n_x_, n_x_);;

    P_pred.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
    {  
      VectorXd x_diff = Xsig_pred_.col(i) - state;
      x_diff(3) = normalizeAngle(x_diff(3));

      P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose();
    }

    return P_pred;
  }



MatrixXd UKF::computeMeasurementCovMatrix(MatrixXd &S, MatrixXd &Z_sig, VectorXd &z_pred)
{
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  { 
    VectorXd z_diff = Z_sig.col(i) - z_pred;  
    z_diff(1) = normalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  return S;
}

MatrixXd UKF::computeCrosscorrelationMatrix(MatrixXd &Tc, MatrixXd &Z_sig, VectorXd &z_pred)
{
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  
    VectorXd z_diff = Z_sig.col(i) - z_pred;
    z_diff(1) = normalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  return Tc;
}

VectorXd UKF::computeMeanPredictedMeasurement(VectorXd &z_pred, MatrixXd &Z_sig)
{
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) 
  {
    z_pred = z_pred + weights_(i) * Z_sig.col(i);
  }
  return z_pred;
}

void UKF::updateState(MatrixXd &Tc, MatrixXd &S, VectorXd &z, VectorXd &z_pred)
{
  MatrixXd K = Tc * S.inverse(); // Kalman gain K

  VectorXd z_diff = z - z_pred;  // residual
  
  z_diff(1) = normalizeAngle(z_diff(1));

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}