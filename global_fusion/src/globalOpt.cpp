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

#include "globalOpt.h"
#include "Factors.h"

GlobalOptimization::GlobalOptimization()
{
    initGPS = false;
    newGPS = false;
    // 首先给的一个单位矩阵，用于初始化，后期会优化
    WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
}

GlobalOptimization::~GlobalOptimization()
{
    threadOpt.detach();
}

// 用于将GPS坐标（经度、纬度、高程）转换为三维笛卡尔坐标系（x, y, z）
void GlobalOptimization::GPS2XYZ(double latitude, double longitude, double altitude, double *xyz)
{
    if (!initGPS)
    {
        geoConverter.Reset(latitude, longitude, altitude);
        initGPS = true;
    }
    geoConverter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);
    // printf("la: %f lo: %f al: %f\n", latitude, longitude, altitude);
    // printf("gps x: %f y: %f z: %f\n", xyz[0], xyz[1], xyz[2]);
}

// 将VIO输出的位姿转换到GPS坐标系下放入到globalPoseMap 中，并且将vio输出的位姿存入localPoseMap 中
void GlobalOptimization::inputOdom(double t, Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ)
{
    mPoseMap.lock();
    // 把vio直接输出的位姿存入 localPoseMap 中
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(),
                             OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;

    // 把VIO转换到GPS坐标系下，准确的说是转换到以第一帧GPS为原点的坐标系下
    // 转换之后的位姿插入到globalPoseMap 中
    Eigen::Quaterniond globalQ;
    globalQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP + WGPS_T_WVIO.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    lastP = globalP;
    lastQ = globalQ;

    // 把vio转换坐标系后的结果赋值给global_path，给最新传入的一个初始值。
    // 当前的vio也会加入到global_path 中，后续收到GPS数据时，会将其全部删除之后，进行更新，所以有时候会看到融合之后的路径发生跳变的情况
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(t);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = lastP.x();
    pose_stamped.pose.position.y = lastP.y();
    pose_stamped.pose.position.z = lastP.z();
    pose_stamped.pose.orientation.x = lastQ.x();
    pose_stamped.pose.orientation.y = lastQ.y();
    pose_stamped.pose.orientation.z = lastQ.z();
    pose_stamped.pose.orientation.w = lastQ.w();
    global_path.header = pose_stamped.header;
    global_path.poses.push_back(pose_stamped);

    mPoseMap.unlock();
}

// 获取最新的位姿lastP和lastQ
void GlobalOptimization::getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

// 将GPS数据转到平面坐标系并且以第一个GPS数据为原点，放入到GPSPositionMap 中
void GlobalOptimization::inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy)
{
    double xyz[3];
    // 将GPS的经纬度转换到ENU坐标系
    // 因为经纬度表示的是地球上的坐标，而地球是一个球形，
    // 需要首先把经纬度转化到平面坐标系上
    // 值得一提的是，GPS2XYZ()并非把经纬度转化到世界坐标系下(以0经度，0纬度为原点)，
    // 而是以第一帧GPS数据为坐标原点，这一点需要额外注意
    GPS2XYZ(latitude, longitude, altitude, xyz);
    // 存入经纬度计算出的平面坐标，存入GPSPositionMap中
    vector<double> tmp{xyz[0], xyz[1], xyz[2], posAccuracy};
    // printf("new gps: t: %f x: %f y: %f z:%f \n", t, tmp[0], tmp[1], tmp[2]);
    GPSPositionMap[t] = tmp;
    // 每次获取到新的GPS数据时，设置为true，表示可以开始优化， optimize函数是while循环，每次判断newGPS为true时就会优化
    newGPS = true;
}

void GlobalOptimization::optimize()
{
    while (true)
    {
        if (newGPS)
        {
            newGPS = false;
            printf("global optimization\n");
            TicToc globalOptimizationTime;
            // 定义优化问题和求解器
            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            // options.minimizer_progress_to_stdout = true;
            // options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(1.0);
            ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

            // add param
            mPoseMap.lock();
            int length = localPoseMap.size();
            // w^t_i   w^q_i
            // 将vio数据作为优化参数添加
            double t_array[length][3];
            double q_array[length][4];
            map<double, vector<double>>::iterator iter;
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++)
            {
                t_array[i][0] = iter->second[0];
                t_array[i][1] = iter->second[1];
                t_array[i][2] = iter->second[2];
                q_array[i][0] = iter->second[3];
                q_array[i][1] = iter->second[4];
                q_array[i][2] = iter->second[5];
                q_array[i][3] = iter->second[6];
                // local_parameterization 是一个参数化对象，用于处理四元数的优化问题，确保四元数保持单位长度
                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
            }

            map<double, vector<double>>::iterator iterVIO, iterVIONext, iterGPS;
            int i = 0;
            for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++)
            {
                // vio factor
                iterVIONext = iterVIO;
                iterVIONext++;
                // 添加VIO残差，观测量是两帧VIO数据之差，是相对的。而下面的GPS是绝对的
                if (iterVIONext != localPoseMap.end())
                {
                    Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();
                    wTi.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4],
                                                               iterVIO->second[5], iterVIO->second[6])
                                                .toRotationMatrix();
                    wTi.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    wTj.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4],
                                                               iterVIONext->second[5], iterVIONext->second[6])
                                                .toRotationMatrix();
                    wTj.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                    Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                    Eigen::Quaterniond iQj;
                    // 提取相对变换矩阵iTj的旋转部分，得到相对旋转四元数iQj
                    iQj = iTj.block<3, 3>(0, 0);
                    // 提取相对变换矩阵iTj的平移部分，得到相对平移向量iPj
                    Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);
                    // 创建一个Ceres优化的CostFunction对象，这个对象是用于计算相对运动误差的代价函数，其中传入了相对平移和相对旋转以及对应的权重。
                    ceres::CostFunction *vio_function = RelativeRTError::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                                iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                                0.1, 0.01);
                    // 将上述创建的代价函数添加到Ceres优化问题中作为残差块。这里使用了四元数和平移向量作为优化变量
                    // NULL表示使用默认的平方损失函数，这里就可以理解要使得优化之后的vio的数据更加平滑或者是均等，因为只有当每一段的
                    // 误差基本相等的时候，才会达到最小的平方误差。
                    // vio_function在创建时使用了localPoseMap中相邻两帧见的平移误差和四元数误差，然后添加残差块的时候，使用globalPoseMap中
                    // 相邻的t_array和q_array来求平移误差和四元数误差，最后将这两个对应的误差分别相减来构造最后的残差。
                    problem.AddResidualBlock(vio_function, NULL, q_array[i], t_array[i], q_array[i + 1], t_array[i + 1]);

                    /*
                    double **para = new double *[4];
                    para[0] = q_array[i];
                    para[1] = t_array[i];
                    para[3] = q_array[i+1];
                    para[4] = t_array[i+1];

                    double *tmp_r = new double[6];
                    double **jaco = new double *[4];
                    jaco[0] = new double[6 * 4];
                    jaco[1] = new double[6 * 3];
                    jaco[2] = new double[6 * 4];
                    jaco[3] = new double[6 * 3];
                    vio_function->Evaluate(para, tmp_r, jaco);

                    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 1>>(tmp_r).transpose() << std::endl
                        << std::endl;
                    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>>(jaco[0]) << std::endl
                        << std::endl;
                    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>>(jaco[1]) << std::endl
                        << std::endl;
                    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>>(jaco[2]) << std::endl
                        << std::endl;
                    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>>(jaco[3]) << std::endl
                        << std::endl;
                    */
                }
                // gps factor
                double t = iterVIO->first; // t表示的是时间
                // 在GPSPositionMap中查找是否有与当前时间戳相匹配的GPS数据。GPSPositionMap是一个映射，它将时间戳映射到对应的GPS位置信息
                // GPS残差，这个观测量直接就是GPS的测量数据，
                // 残差计算的是GPS和优化变量的差，这个是绝对的差。
                iterGPS = GPSPositionMap.find(t);

                if (iterGPS != GPSPositionMap.end())
                {
                    // 根据找到的GPS数据创建一个Ceres优化的CostFunction对象。这个对象用于计算VIO估计的轨迹与GPS轨迹之间的误差。
                    // GPS位置信息通常由经度、纬度、高度和权重（或协方差）组成，这里使用了这些信息创建了一个代价函数
                    ceres::CostFunction *gps_function = TError::Create(iterGPS->second[0], iterGPS->second[1],
                                                                       iterGPS->second[2], iterGPS->second[3]);
                    // printf("inverse weight %f \n", iterGPS->second[3]);
                    // 将上述创建的代价函数添加到Ceres优化问题中作为残差块。这个残差块是用于优化VIO轨迹，
                    // 它将VIO估计的轨迹与GPS轨迹进行比较，并通过最小化它们之间的差异来调整VIO估计的结果。t_array[i]是VIO优化变量，它表示时间戳对应的位姿。
                    // Ceres求解器会根据代价函数和损失函数，调整这个t_array参数块中的值以最小化残差。

                    // 下面gps_function定义了如何使用t_array[i]计算GPS轨迹的残差，gps_function是一个仿函数，在创建它的时候，
                    // 使用了iterGPS->second[0], iterGPS->second[1], iterGPS->second[2], iterGPS->second[3]来创建了，就相当于
                    // 在内部维护了三个值，这个值将来是要被t_array来减去的，就形成了vio和GPS的残差，因为t_array是来自于globalPoseMap的，
                    // globalPoseMap是未经优化的vio的数据，那么这个代码的含义就是要最小化对应的GPS和vio的位置的距离，
                    // loss_function是一个Huber损失函数，用于调整距离误差的大小，初始化时设置的是1.0，如果误差小于等于1.0，就使用默认的平方误差，
                    // 如果损失超过了1.0，就使用Huber损失定义的线性的误差（没有了平方的效果），相当于降低了损失的贡献程度，在优化过程中，对异常值的
                    // 处理就显得更加温和，降低了异常值对总误差的影响。

                    // 使用 HuberLoss 的具体作用是：
                    // 使优化过程对异常值（outliers）具有更好的鲁棒性。
                    // 对小残差进行平方处理（与标准L2范数相同），对大残差进行线性处理，从而减小异常值对总误差的影响。
                    // 提高优化结果的稳健性，使得最终结果不容易被少量异常值所极大影响。
                    problem.AddResidualBlock(gps_function, loss_function, t_array[i]);

                    /*
                    double **para = new double *[1];
                    para[0] = t_array[i];

                    double *tmp_r = new double[3];
                    double **jaco = new double *[1];
                    jaco[0] = new double[3 * 3];
                    gps_function->Evaluate(para, tmp_r, jaco);

                    std::cout << Eigen::Map<Eigen::Matrix<double, 3, 1>>(tmp_r).transpose() << std::endl
                        << std::endl;
                    std::cout << Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(jaco[0]) << std::endl
                        << std::endl;
                    */
                }
            }
            // mPoseMap.unlock();
            ceres::Solve(options, &problem, &summary);
            // std::cout << summary.BriefReport() << "\n";

            // update global pose
            // mPoseMap.lock();
            // 经过上面的优化之后，t_array和q_array中的数据就是优化后的全局位姿
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++)
            {
                // 从VIO优化的位姿数据中提取位置和四元数，构建一个包含全局位姿的向量globalPose
                vector<double> globalPose{t_array[i][0], t_array[i][1], t_array[i][2],
                                          q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
                // 更新全局位姿映射中当前时间戳对应的值为新的全局位姿
                iter->second = globalPose;
                // 如果已经遍历到最后一个位姿数据
                // 用在VIO坐标系下的位置（localPoseMap里）和优化后在GPS坐标系下的位置（globalPose）对外参WGPS_T_WVIO进行更新
                if (i == length - 1)
                {
                    Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                    // 获取当前时间戳
                    double t = iter->first;
                    // 从localPoseMap获取VIO相机坐标系到世界坐标系的旋转矩阵
                    WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4],
                                                                       localPoseMap[t][5], localPoseMap[t][6])
                                                        .toRotationMatrix();
                    // 从localPoseMap获取VIO相机坐标系到世界坐标系的平移向量
                    WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                    // 从globalPose获取GPS到世界坐标系的旋转矩阵
                    WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4],
                                                                       globalPose[5], globalPose[6])
                                                        .toRotationMatrix();
                    // 从globalPose获取GPS到世界坐标系的平移向量
                    WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                    // 计算GPS到VIO的变换，即WGPS_T_WVIO
                    // 外参WGPS_T_WVIO之前是单位矩阵，而第一次更新，会算出真正和实际相符的外参
                    WGPS_T_WVIO = WGPS_T_body * WVIO_T_body.inverse();
                }
            }
            updateGlobalPath();
            // printf("global time %f \n", globalOptimizationTime.toc());
            mPoseMap.unlock();
        }
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
    return;
}

// 清空global_path.poses，重新根据globalPoseMap生成
void GlobalOptimization::updateGlobalPath()
{
    // 清空全局路径中的所有位姿
    global_path.poses.clear();
    // 定义一个迭代器iter，用于遍历globalPoseMap
    map<double, vector<double>>::iterator iter;
    // 优化之后，globalPoseMap被重现填入了优化后的数据，就是那些被优化了的t_array, q_array数据
    for (iter = globalPoseMap.begin(); iter != globalPoseMap.end(); iter++)
    {
        // 定义一个geometry_msgs::PoseStamped类型的变量pose_stamped，用于存储位姿信息
        geometry_msgs::PoseStamped pose_stamped;
        // 设置pose_stamped的时间戳，使用globalPoseMap中的时间戳（迭代器iter的键）
        pose_stamped.header.stamp = ros::Time(iter->first);
        // 设置pose_stamped的坐标系为“world”
        pose_stamped.header.frame_id = "world";
        // 设置pose_stamped的位置信息，使用globalPoseMap中的x坐标（迭代器iter的值的第一个元素）
        pose_stamped.pose.position.x = iter->second[0];
        pose_stamped.pose.position.y = iter->second[1];
        pose_stamped.pose.position.z = iter->second[2];
        pose_stamped.pose.orientation.w = iter->second[3];
        pose_stamped.pose.orientation.x = iter->second[4];
        pose_stamped.pose.orientation.y = iter->second[5];
        pose_stamped.pose.orientation.z = iter->second[6];
        // 将构建好的pose_stamped添加到global_path的poses序列中
        global_path.poses.push_back(pose_stamped);
    }
}