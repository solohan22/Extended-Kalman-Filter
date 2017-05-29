#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_radar = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
    */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    //state covariance matrix P
    ekf_.P_ = MatrixXd (4,4);
    ekf_.P_ << 1,0,0,0,
               0,1,0,0,
               0,0,1000,0,
               0,0,0,1000;

    // lidar measurement matrix
    H_laser_ << 1,0,0,0,
                0,1,0,0;

    // radar measurement matrix
    Hj_radar << 1,1,0,0,
                      1,1,0,0,
                      1,1,1,1;

    // the initial transition matrix F_
    ekf_.F_ = MatrixXd(4,4);
    ekf_.F_ << 1,0,1,0,
                     0,1,0,1,
                     0,0,1,0,
                     0,0,0,1;

    //set the acceleration noise components
    noise_ax = 9;
    noise_ay = 9;

    // the initial timestamp 
    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      if sensor is radar, convert radar from polar to cartesian coordinates and initialize state
      */
      if (measurement_pack.raw_measurements_[0] == 0 or measurement_pack.raw_measurements_[1] == 0){
        ekf_.x_<< 0, 0, 0, 0;
        return;
      }
      
      // conversion formula: px = rho * cos(phi), py = rho * sin(phi), vx = rho_dot * cos(phi), vy = rho_dot * sin(phi)
      ekf_.x_ <<measurement_pack.raw_measurements_[0]*cos(measurement_pack.raw_measurements_[1]), 
                  measurement_pack.raw_measurements_[0]*sin(measurement_pack.raw_measurements_[1]), 
                  measurement_pack.raw_measurements_[2]*cos(measurement_pack.raw_measurements_[1]), 
                  measurement_pack.raw_measurements_[2]*sin(measurement_pack.raw_measurements_[1]);   
    }

    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      if sensor is lidar, directly initialize state
      */// 
      if (measurement_pack.raw_measurements_[0] == 0 or measurement_pack.raw_measurements_[1] == 0){
        return;
      }
      ekf_.x_<<measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // initialization done
    is_initialized_ = true;
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time. Time is measured in seconds.
     * Update the process noise covariance matrix.
   */

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //set the transision matrix F
  ekf_.F_ << 1, 0, dt, 0,
                   0, 1, 0, dt,
                   0, 0, 1, 0,
                   0, 0, 0, 1;
  
  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
                   0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
                   dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
                   0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  // call Predict() of EKF instance
  ekf_.Predict();


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // radar updates
    Hj_radar << tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_radar;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);  
  } 
  else {
    // lidar updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}