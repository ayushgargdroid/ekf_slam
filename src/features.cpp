#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/LaserScan.h"

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

void chatterCallback(const sensor_msgs::LaserScan _msg)
{
  comm = _msg;
  // ROS_INFO("Frame ID: %s\nAngle Max: %f Min: %f\nAngle Increment: %f\nTime increment: %f\nScan Time: %f\nRange Max: %f Min: %f\nLen: %ld\n",_msg.header.frame_id.c_str(),_msg.angle_max,_msg.angle_min,_msg.angle_increment,_msg.time_increment,_msg.scan_time,_msg.range_max,_msg.range_min,_msg.intensities.size());
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_extractor");

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("/hsrb/base_scan", 1, chatterCallback);
  ros::Publisher chatter_pub = n.advertise<sensor_msgs::LaserScan>("/hsrb/filter_scan", 10);

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
  falkolib::NNMatcher<falkolib::BSC> matcher2;
  matcher.setDistanceThreshold(0.1);
  std::vector<std::pair<int, int> > assoNN;
  std::vector<std::pair<int, int> > assoNN2;
  std::vector<std::vector<falkolib::FALKO>> keypoints;
  std::vector<std::vector<falkolib::BSC>> discriptors;

  int iter = 0;
  while(ros::ok())
  {
    ros::spinOnce();

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
      // ROS_INFO("Matcher: %d",matcher.match(keypoints[iter-1],keypoints[iter],assoNN));
      // ROS_INFO("Keypoints1 size:%ld Keypoints2 size:%ld Keypoints size:%ld",keypoints[iter-2].size(),keypoints[iter-1].size(),keypoints.size());
      ROS_INFO("Matcher1: %d",matcher.match(keypoints[iter-2],keypoints[iter-1],assoNN));
      ROS_INFO("Matcher2: %d",matcher2.match(discriptors[iter-2],discriptors[iter-1],assoNN2));
      for(int i = 0;i<assoNN.size();i++)
      {
        ROS_INFO("1Matching Pairs %d %d",assoNN[i].first,assoNN[i].second);
      }
      for(int i = 0;i<assoNN2.size();i++)
      {
        ROS_INFO("2Matching Pairs %d %d",assoNN2[i].first,assoNN2[i].second);
      }
    }
    
    // ROS_INFO("BSC Descriptor length: %ld",bscDesc.size());

    for(int i=0;i<keypoints1.size();i++)
    {
      ROS_INFO("Index: %d Radius: %f Orientation: %f",keypoints1[i].index,keypoints1[i].radius,keypoints1[i].orientation);
      new_intensity[keypoints1[i].index] = 1.0;
    }
    msg.intensities = new_intensity;
    chatter_pub.publish(msg);

    ROS_INFO("Frame ID: %s\nAngle Max: %f Min: %f\nAngle Increment: %f\nTime increment: %f\nScan Time: %f\nRange Max: %f Min: %f\nLen: %ld\n",comm.header.frame_id.c_str(),comm.angle_max,comm.angle_min,comm.angle_increment,comm.time_increment,comm.scan_time,comm.range_max,comm.range_min,comm.intensities.size());

    iter++;
    keypoints.push_back(keypoints1);
    discriptors.push_back(bscDesc);
    r.sleep();
  }

  return 0;
}