#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Pose2D.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>

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
  moveCmd.angular.z = 0.1;

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
  std::vector<std::vector<falkolib::FALKO>> keypoints;
  std::vector<std::vector<falkolib::BSC>> discriptors;

  int N = 100;
  int landmarkSize = 0;

  Eigen::MatrixXd meanMat(3+2*landmarkSize,1);
  Eigen::MatrixXd covMat(3+2*landmarkSize,3+2*landmarkSize);
  Eigen::MatrixXd R(3,3);
  Eigen::MatrixXd G(3+2*landmarkSize,3+2*landmarkSize);
  Eigen::MatrixXd Fx(3,3+2*landmarkSize);
  Eigen::MatrixXd Q(3,3);

  Fx = Eigen::MatrixXd::Zero(3,3+2*landmarkSize);
  covMat = Eigen::MatrixXd::Zero(3+2*landmarkSize,3+2*landmarkSize);
  R = Eigen::MatrixXd::Zero(3,3);
  Q = Eigen::MatrixXd::Zero(2,2);
  // Eigen::RowVectorXd t(3+2*landmarkSize);
  // t.fill(10);
  // t(0) = t(1) = t(2) = 0;
  // covMat += t.asDiagonal();
  Fx(0,0) = Fx(1,1) = Fx(2,2) = 1;
  R(0,0) = R(1,1) = R(2,2) = 0.5;
  Q(0,0) = Q(1,1) = 1.0;

  ros::spinOnce();

  meanMat(0,0) = currentPose.x;
  meanMat(1,0) = currentPose.y;
  meanMat(2,0) = currentPose.theta;

  std::ofstream myfile1;
  myfile1.open ("output.txt");

  std::ofstream myfile2;
  myfile2.open ("output1.txt");

  int iter = 0;
  while(ros::ok())
  {
    ros::spinOnce();

    Eigen::MatrixXd t1(3,1);
    t1(0,0) = -(moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0)) + (moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0) + moveCmd.angular.z*0.1);
    t1(1,0) = (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0)) - (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0) + moveCmd.angular.z*0.1);
    t1(2,0) = moveCmd.angular.z*0.1;
    Eigen::MatrixXd t2(3,3);
    t2 = Eigen::MatrixXd::Zero(3,3);
    t2(0,2) = (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0)) - (moveCmd.linear.x/moveCmd.angular.z)*cos(meanMat(2,0) + moveCmd.angular.z*0.1);
    t2(1,2) = (moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0)) - (moveCmd.linear.x/moveCmd.angular.z)*sin(meanMat(2,0) + moveCmd.angular.z*0.1);

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

    if(iter>=2)
    {
      for(int i = 0;i<iter;i++)
      {
        int tt = matcher.match(keypoints1,keypoints[i],assoNN);
        int t2312 = matcher2.match(bscDesc,discriptors[i],assoNN2);
        // ROS_INFO("Matcher: %d %d",i,t2312);
        myfile1<<"Matcher: "<<i<<" "<<t2312<<"\n";
        myfile2<<"Matcher: "<<i<<" "<<tt<<"\n";
        // ROS_INFO("Matching Pairs %d %d",assoNN2[i].first,assoNN2[i].second);
      }
      myfile1<<"\n";
      myfile2<<"\n";
    }

    for(int i=0;i<keypoints1.size();i++)
    {
      ROS_INFO("Index: %d Radius: %f Orientation: %f",keypoints1[i].index,keypoints1[i].radius,keypoints1[i].orientation);
      new_intensity[keypoints1[i].index] = 1.0;
    }

    ROS_INFO("BSC Descriptor length: %ld",bscDesc.size());

    msg.intensities = new_intensity;
    filterScanPub.publish(msg);

    ROS_INFO("Frame ID: %s\nAngle Max: %f Min: %f\nAngle Increment: %f\nTime increment: %f\nScan Time: %f\nRange Max: %f Min: %f\nLen: %ld\n",comm.header.frame_id.c_str(),comm.angle_max,comm.angle_min,comm.angle_increment,comm.time_increment,comm.scan_time,comm.range_max,comm.range_min,comm.intensities.size());

    iter++;
    keypoints.push_back(keypoints1);
    discriptors.push_back(bscDesc);
    r.sleep();
  }

  myfile1.close();
  myfile2.close();

  return 0;
}