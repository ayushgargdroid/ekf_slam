#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Pose2D.h"

#include <Eigen/Dense>

#include <falkolib/Feature/FALKO.h>
#include <falkolib/Feature/CGH.h>
#include <falkolib/Feature/BSC.h>
#include <falkolib/Feature/FALKOExtractor.h>
#include <falkolib/Feature/BSCExtractor.h>

#include <falkolib/Matching/NNMatcher.h>

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */

sensor_msgs::LaserScan comm;
geometry_msgs::Pose2D currentPose;

void laserScanCallback(const sensor_msgs::LaserScan _msg)
{
  comm = _msg;
  // ROS_INFO("Frame ID: %s\nAngle Max: %f Min: %f\nAngle Increment: %f\nTime increment: %f\nScan Time: %f\nRange Max: %f Min: %f\nLen: %ld\n",_msg.header.frame_id.c_str(),_msg.angle_max,_msg.angle_min,_msg.angle_increment,_msg.time_increment,_msg.scan_time,_msg.range_max,_msg.range_min,_msg.intensities.size());
}

void poseCallback(geometry_msgs::Pose2D _msg)
{
  currentPose = _msg;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_extractor");

  ros::NodeHandle n;

  ros::Subscriber laserScanSub = n.subscribe("/hsrb/base_scan", 1, laserScanCallback);
  ros::Subscriber poseSub = n.subscribe("/hsrb/Pose2D", 1, poseCallback);

  ros::Publisher filterScanPub = n.advertise<sensor_msgs::LaserScan>("/hsrb/filter_scan", 10);
  ros::Publisher moveCmdPub = n.advertise<geometry_msgs::Twist>("/hsrb/command_velocity",10);

  geometry_msgs::Twist moveCmd;
  moveCmd.angular.z = moveCmd.linear.x = 0;

  ros::Rate(1).sleep();
  ros::Rate r(10);

  falkolib::FALKOExtractor fe;
  fe.setMinExtractionRange(1);
  fe.setMaxExtractionRange(30);
  fe.enableSubbeam(true);
  fe.setNMSRadius(0.1);
  fe.setNeighB(0.07);
  fe.setBRatio(2.5);
  fe.setGridSectors(16);

  falkolib::BSCExtractor<falkolib::FALKO> bsc(16,8);
  falkolib::NNMatcher<falkolib::FALKO> matcher;
  matcher.setDistanceThreshold(0.1);
  falkolib::NNMatcher<falkolib::BSC> matcher2;
  std::vector<std::pair<int, int> > assoNN;
  std::vector<std::pair<int, int> > assoNN2;
  std::vector<falkolib::FALKO> keypoints;
  std::vector<falkolib::BSC> discriptors;

  int N = 100;
  int landmarkSize = 0;

  Eigen::MatrixXd meanMat(3+2*landmarkSize,1);
  Eigen::MatrixXd covMat(3+2*landmarkSize,3+2*landmarkSize);
  Eigen::MatrixXd R(3,3);
  Eigen::MatrixXd G(3+2*landmarkSize,3+2*landmarkSize);
  Eigen::MatrixXd Q(2,2);
  Eigen::MatrixXd Z(2,1);

  
  covMat = Eigen::MatrixXd::Zero(3+2*landmarkSize,3+2*landmarkSize);
  R = Eigen::MatrixXd::Zero(3,3);
  Q = Eigen::MatrixXd::Zero(2,2);
  // Eigen::RowVectorXd t(3+2*landmarkSize);
  // t.fill(10);
  // t(0) = t(1) = t(2) = 0;
  // covMat += t.asDiagonal(); 
  R(0,0) = R(1,1) = R(2,2) = 0.5;
  Q(0,0) = Q(1,1) = 1.0;

  ros::spinOnce();

  meanMat(0,0) = currentPose.x;
  meanMat(1,0) = currentPose.y;
  meanMat(2,0) = currentPose.theta;

  int iter = 0;
  while(ros::ok())
  {
    ros::spinOnce();

    Eigen::MatrixXd t1(3,1);
    Eigen::MatrixXd Fx(3,3+2*landmarkSize);
    
    Fx = Eigen::MatrixXd::Zero(3,3+2*landmarkSize);
    Fx(0,0) = Fx(1,1) = Fx(2,2) = 1;

    t1(0,0) = -(moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0)) + (moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0) + moveCmd.angular.z*0.1);
    t1(1,0) = (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0)) - (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0) + moveCmd.angular.z*0.1);
    t1(2,0) = moveCmd.angular.z*0.1;
    Eigen::MatrixXd t2(3,3);
    t2 = Eigen::MatrixXd::Zero(3,3);
    t2(0,2) = (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0)) - (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0) + moveCmd.angular.z*0.1);
    t2(1,2) = (moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0)) - (moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0) + moveCmd.angular.z*0.1);
    ROS_INFO("Mean Mat: %d %d",meanMat.rows(),meanMat.cols());

    meanMat = meanMat + Fx.transpose() * t1;
    
    G = Eigen::MatrixXd::Identity(3+2*landmarkSize,3+2*landmarkSize) + Fx.transpose() * t2 * Fx;

    covMat = (G * covMat * G.transpose()) + (Fx.transpose() * R * Fx);

    sensor_msgs::LaserScan msg;

    msg = comm;

    falkolib::LaserScan scan1(comm.angle_min,(comm.angle_max - comm.angle_min),721);
    std::vector<double> doubleRange(comm.ranges.begin(),comm.ranges.end());
    scan1.fromRanges(doubleRange);

    std::vector<float> new_intensity(721,0);
    std::vector<falkolib::FALKO> keypoints1;
    std::vector<falkolib::BSC> bscDesc;

    fe.extract(scan1,keypoints1);
    bsc.compute(scan1,keypoints1,bscDesc);
    
    // int matches = matcher.match(keypoints,keypoints1,assoNN);
    std::cout<<"Mean mat:\n"<<meanMat.block(0,0,3,1)<<std::endl;
    
    ROS_INFO("Total keypoints detected: %ld",keypoints1.size());

    for(int i=0;i<keypoints1.size();i++)
    {
      ROS_INFO("Index: %d Radius: %f Orientation: %f",keypoints1[i].index,keypoints1[i].radius,keypoints1[i].orientation);
      meanMat.conservativeResize(3+2*(landmarkSize+1),1);
      meanMat(2*landmarkSize+3,0) = meanMat(0) + doubleRange[keypoints1[i].index] * cos(keypoints1[i].orientation + meanMat(2,0));
      meanMat(2*landmarkSize+3+1,0) = meanMat(1) + doubleRange[keypoints1[i].index] * sin(keypoints1[i].orientation + meanMat(2,0));
      Z(0,0) = doubleRange[keypoints1[i].index];
      Z(1,0) = keypoints1[i].orientation; 
      
      covMat.conservativeResize(3+2*(landmarkSize+1),3+2*(landmarkSize+1));
      covMat.block(0,3+2*landmarkSize,2*(landmarkSize+1)+3,2) = Eigen::MatrixXd::Zero(3+2*(landmarkSize+1),2);
      covMat.block(3+2*landmarkSize,0,2,2*landmarkSize+3) = Eigen::MatrixXd::Zero(2,3+2*landmarkSize);
      covMat(3+2*landmarkSize,3+2*landmarkSize) = covMat(3+1+2*landmarkSize,3+1+2*landmarkSize) = 10.0;
      
      new_intensity[keypoints1[i].index] = 1.0;

      Eigen::MatrixXd hList[landmarkSize+1];
      Eigen::MatrixXd zetaList[landmarkSize+1];
      Eigen::MatrixXd zList[landmarkSize+1];
      double dist;
      std::vector<double> distList;

      Eigen::MatrixXd H(2,2*(landmarkSize+1)+3);
      Eigen::MatrixXd K(2*(landmarkSize+1)+3,2);
      Eigen::MatrixXd t(2,5);
      Eigen::MatrixXd delta(2,1);
      Eigen::MatrixXd z(2,1);
      Eigen::MatrixXd Zeta(2,2);

      Zeta = Eigen::MatrixXd::Zero(2,2);
      for(int k = 0; k<(landmarkSize+1);k++)
      {
        delta(0,0) = meanMat(2*k+3,0) - meanMat(0,0);
        delta(1,0) = meanMat(2*k+3+1,0) - meanMat(1,0);
        
        double q = pow(delta(0,0),2) + pow(delta(1,0),2);
        
        z(0,0) = sqrt(q);
        z(1,0) = atan2(delta(1,0),delta(0,0)) - meanMat(2,0);
        // std::cout<<"z"<<k<<":\n"<<z<<std::endl;
        zList[k] = z;

        Eigen::MatrixXd Fxk(5,2*(landmarkSize+1)+3);
        Fxk = Eigen::MatrixXd::Zero(5,2*(landmarkSize+1)+3);
        Fxk(0,0) = Fxk(1,1) = Fxk(2,2) = 1;
        Fxk(3,2*k+3) = Fxk(4,2*k+1+3) = 1;
        // std::cout<<"Fxk:\n"<<Fxk<<std::endl;

        t(0,0) = sqrt(q) * delta(0,0);
        t(0,1) = -sqrt(q) * delta(1,0);
        t(0,2) = 0;
        t(0,3) = -sqrt(q) * delta(0,0);
        t(0,4) = sqrt(q) * delta(1,0);
        t(1,0) = delta(1,0);
        t(1,1) = delta(0,0);
        t(1,2) = -1;
        t(1,3) = -delta(1,0);
        t(1,4) = -delta(0,0);
        // std::cout<<"t:\n"<<t<<std::endl;

        H = t * Fxk/q;
        // std::cout<<"H:\n"<<H<<std::endl;
        hList[k] = H;

        Zeta = H * covMat * H.transpose() + Q;
        zetaList[k] = Zeta;

        // ROS_INFO("zi: %ld %ld zk: %ld %ld Zeta: %ld %ld",Z.rows(),Z.cols(),zList[k].rows(),zList[k].cols(),Zeta.rows(),Zeta.cols());
        dist = ((Z - zList[k]).transpose() * Zeta.inverse() * (Z - zList[k]))(0);
        // std::cout<<"Distance: "<<dist<<std::endl;
        distList.push_back(dist);
      }
      distList[landmarkSize] = 0.1;
      // std::cout<<"All distances:\n";
      // for(int i=0;i<distList.size();i++)
      //   std::cout<<i<<" "<<distList[i]<<std::endl;
      int argMin = std::min_element(distList.begin(),distList.end()) - distList.begin();
      landmarkSize = std::max(landmarkSize,argMin+1);
      // std::cout<<"landmark size: "<<landmarkSize<<" argmin: "<<argMin<<std::endl;
      if(landmarkSize != argMin-1)
      {
        meanMat.conservativeResize(3+2*(landmarkSize),1);
        covMat.conservativeResize(3+2*(landmarkSize),3+2*(landmarkSize));
        hList[argMin].conservativeResize(2,3+2*landmarkSize);
        // std::cout<<"Reduced mean mat:\n"<<meanMat<<std::endl;
      }
      else
      {
        keypoints.push_back(keypoints1[i]);
      }

      K = covMat * hList[argMin].transpose() * zetaList[argMin].inverse();
      meanMat = meanMat + K*(Z - zList[argMin]);
      covMat = (Eigen::MatrixXd::Identity(3+2*landmarkSize,3+2*landmarkSize) - K * hList[argMin])*covMat;
      ROS_INFO("Keypoint size: %ld",landmarkSize);
    }

    msg.intensities = new_intensity;
    filterScanPub.publish(msg);

    // ROS_INFO("Frame ID: %s\nAngle Max: %f Min: %f\nAngle Increment: %f\nTime increment: %f\nScan Time: %f\nRange Max: %f Min: %f\nLen: %ld\n",comm.header.frame_id.c_str(),comm.angle_max,comm.angle_min,comm.angle_increment,comm.time_increment,comm.scan_time,comm.range_max,comm.range_min,comm.intensities.size());

    iter++;
    // if(iter == 3)
    //   break;
    r.sleep();
  }

  return 0;
}