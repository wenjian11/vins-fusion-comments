/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "ros/ros.h"
#include "globalOpt.h"
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <queue>
#include <mutex>

GlobalOptimization globalEstimator;
ros::Publisher pub_global_odometry, pub_global_path, pub_car;
nav_msgs::Path *global_path;
double last_vio_t = -1;
std::queue<sensor_msgs::NavSatFixConstPtr> gpsQueue;
std::mutex m_buf;

// publish car model
void publish_car_model(double t, Eigen::Vector3d t_w_car, Eigen::Quaterniond q_w_car)
{
    visualization_msgs::MarkerArray markerArray_msg;
    visualization_msgs::Marker car_mesh;
    car_mesh.header.stamp = ros::Time(t);
    car_mesh.header.frame_id = "world";
    car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::Marker::ADD;
    car_mesh.id = 0;

    car_mesh.mesh_resource = "package://global_fusion/models/car.dae";

    Eigen::Matrix3d rot;
    rot << 0, 0, -1, 0, -1, 0, -1, 0, 0;

    Eigen::Quaterniond Q;
    Q = q_w_car * rot;
    car_mesh.pose.position.x = t_w_car.x();
    car_mesh.pose.position.y = t_w_car.y();
    car_mesh.pose.position.z = t_w_car.z();
    car_mesh.pose.orientation.w = Q.w();
    car_mesh.pose.orientation.x = Q.x();
    car_mesh.pose.orientation.y = Q.y();
    car_mesh.pose.orientation.z = Q.z();

    car_mesh.color.a = 1.0;
    car_mesh.color.r = 1.0;
    car_mesh.color.g = 0.0;
    car_mesh.color.b = 0.0;

    float major_scale = 2.0;

    car_mesh.scale.x = major_scale;
    car_mesh.scale.y = major_scale;
    car_mesh.scale.z = major_scale;
    markerArray_msg.markers.push_back(car_mesh);
    pub_car.publish(markerArray_msg);
}

// 向gpsQueue中插入GPS_msg数据
void GPS_callback(const sensor_msgs::NavSatFixConstPtr &GPS_msg)
{
    // printf("gps_callback! \n");
    m_buf.lock();
    gpsQueue.push(GPS_msg);
    m_buf.unlock();
}

// 完成了vio和GPS数据的融合，并且发布全局位姿，将结果存入文件中
void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    // printf("vio_callback! \n");
    double t = pose_msg->header.stamp.toSec();
    last_vio_t = t;
    // 获取VIO输出的位置(三维向量),姿态(四元数)
    Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    // vio_q表示姿态数据
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;
    // 位姿传入global Estimator中
    // 下面这个函数完成了三个步骤
    // (1) 将vio_t和vio_q合并为localPose数据放入到localPoseMap
    // (2) 将vio_t和vio_q都转为第一帧GPS为原点的坐标系下，然后放入到globalPoseMap中，此时vio是还未优化的数据
    // (3) 然后将当前的vio加入到global_path
    globalEstimator.inputOdom(t, vio_t, vio_q);

    m_buf.lock();
    // 寻找与VIO时间戳相对应的GPS信息，如果找到时间差在10ms内的就进行融合优化，否则一直找到GPS时间戳大于VIO时间戳+10ms时为止
    while (!gpsQueue.empty())
    {
        // 获得最老的GPS数据和其时间
        sensor_msgs::NavSatFixConstPtr GPS_msg = gpsQueue.front();
        double gps_t = GPS_msg->header.stamp.toSec();
        printf("vio t: %f, gps t: %f \n", t, gps_t);
        // 10ms sync tolerance，如果GPS数据在vio时间戳的10ms范围内，就认为是同步的
        if (gps_t >= t - 0.01 && gps_t <= t + 0.01)
        {
            // printf("receive GPS with timestamp %f\n", GPS_msg->header.stamp.toSec());
            // GPS的经纬度,海拔高度
            double latitude = GPS_msg->latitude;
            double longitude = GPS_msg->longitude;
            double altitude = GPS_msg->altitude;
            // int numSats = GPS_msg->status.service;
            // GPS数据的方差
            double pos_accuracy = GPS_msg->position_covariance[0];
            if (pos_accuracy <= 0)
                pos_accuracy = 1;
            // printf("receive covariance %lf \n", pos_accuracy);
            // if(GPS_msg->status.status > 8)
            // 向globalEstimator中输入GPS数据，newGPS会被设置为true，然后优化函数就会执行
            globalEstimator.inputGPS(t, latitude, longitude, altitude, pos_accuracy);
            gpsQueue.pop();
            // 此处break,意味只存储了一个GPS数据后就break了。后来想明白了GPS不同于imu，是绝对位置
            break;
        }
        else if (gps_t < t - 0.01) // GPS数据比vio数据早，就丢掉
            gpsQueue.pop();
        else if (gps_t > t + 0.01) // GPS数据比vio数据晚，就退出，不需要再往后去找10ms内的数据了
            break;
    }
    m_buf.unlock();

    Eigen::Vector3d global_t;
    Eigen::Quaterniond global_q;
    // 目测global_t，global_q是作为传出参数获取vio和GPS融合后的结果的
    globalEstimator.getGlobalOdom(global_t, global_q);
    // 构造里程计数据odometry
    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    // 指定里程计的坐标系
    odometry.header.frame_id = "world";
    // 里程计的子坐标系
    odometry.child_frame_id = "world";
    // 里程计的位姿
    odometry.pose.pose.position.x = global_t.x();
    odometry.pose.pose.position.y = global_t.y();
    odometry.pose.pose.position.z = global_t.z();
    odometry.pose.pose.orientation.x = global_q.x();
    odometry.pose.pose.orientation.y = global_q.y();
    odometry.pose.pose.orientation.z = global_q.z();
    odometry.pose.pose.orientation.w = global_q.w();
    // 发布里程计，这时候发布时最新的位置信息
    pub_global_odometry.publish(odometry);
    // 发布全局路径
    pub_global_path.publish(*global_path);
    // 发布车辆模型到指定的位置，效果就是车模型会根据融合结果进行移动，注意，此时的global_t，global_q是最新的vio数据
    publish_car_model(t, global_t, global_q);

    // write result to file
    std::ofstream foutC("/home/tony-ws1/output/vio_global.csv", ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << pose_msg->header.stamp.toSec() * 1e9 << ",";
    foutC.precision(5);
    foutC << global_t.x() << ","
          << global_t.y() << ","
          << global_t.z() << ","
          << global_q.w() << ","
          << global_q.x() << ","
          << global_q.y() << ","
          << global_q.z() << endl;
    foutC.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "globalEstimator");
    ros::NodeHandle n("~");

    global_path = &globalEstimator.global_path;

    ros::Subscriber sub_GPS = n.subscribe("/gps", 100, GPS_callback);
    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 100, vio_callback);
    pub_global_path = n.advertise<nav_msgs::Path>("global_path", 100);
    pub_global_odometry = n.advertise<nav_msgs::Odometry>("global_odometry", 100);
    pub_car = n.advertise<visualization_msgs::MarkerArray>("car_model", 1000);
    ros::spin();
    return 0;
}
