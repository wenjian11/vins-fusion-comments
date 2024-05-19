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
    // ���ȸ���һ����λ�������ڳ�ʼ�������ڻ��Ż�
    WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
}

GlobalOptimization::~GlobalOptimization()
{
    threadOpt.detach();
}

// ���ڽ�GPS���꣨���ȡ�γ�ȡ��̣߳�ת��Ϊ��ά�ѿ�������ϵ��x, y, z��
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

// ��VIO�����λ��ת����GPS����ϵ�·��뵽globalPoseMap �У����ҽ�vio�����λ�˴���localPoseMap ��
void GlobalOptimization::inputOdom(double t, Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ)
{
    mPoseMap.lock();
    // ��vioֱ�������λ�˴��� localPoseMap ��
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(),
                             OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;

    // ��VIOת����GPS����ϵ�£�׼ȷ��˵��ת�����Ե�һ֡GPSΪԭ�������ϵ��
    // ת��֮���λ�˲��뵽globalPoseMap ��
    Eigen::Quaterniond globalQ;
    globalQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP + WGPS_T_WVIO.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    lastP = globalP;
    lastQ = globalQ;

    // ��vioת������ϵ��Ľ����ֵ��global_path�������´����һ����ʼֵ��
    // ��ǰ��vioҲ����뵽global_path �У������յ�GPS����ʱ���Ὣ��ȫ��ɾ��֮�󣬽��и��£�������ʱ��ῴ���ں�֮���·��������������
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

// ��ȡ���µ�λ��lastP��lastQ
void GlobalOptimization::getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

// ��GPS����ת��ƽ������ϵ�����Ե�һ��GPS����Ϊԭ�㣬���뵽GPSPositionMap ��
void GlobalOptimization::inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy)
{
    double xyz[3];
    // ��GPS�ľ�γ��ת����ENU����ϵ
    // ��Ϊ��γ�ȱ�ʾ���ǵ����ϵ����꣬��������һ�����Σ�
    // ��Ҫ���ȰѾ�γ��ת����ƽ������ϵ��
    // ֵ��һ����ǣ�GPS2XYZ()���ǰѾ�γ��ת������������ϵ��(��0���ȣ�0γ��Ϊԭ��)��
    // �����Ե�һ֡GPS����Ϊ����ԭ�㣬��һ����Ҫ����ע��
    GPS2XYZ(latitude, longitude, altitude, xyz);
    // ���뾭γ�ȼ������ƽ�����꣬����GPSPositionMap��
    vector<double> tmp{xyz[0], xyz[1], xyz[2], posAccuracy};
    // printf("new gps: t: %f x: %f y: %f z:%f \n", t, tmp[0], tmp[1], tmp[2]);
    GPSPositionMap[t] = tmp;
    // ÿ�λ�ȡ���µ�GPS����ʱ������Ϊtrue����ʾ���Կ�ʼ�Ż��� optimize������whileѭ����ÿ���ж�newGPSΪtrueʱ�ͻ��Ż�
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
            // �����Ż�����������
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
            // ��vio������Ϊ�Ż��������
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
                // local_parameterization ��һ���������������ڴ�����Ԫ�����Ż����⣬ȷ����Ԫ�����ֵ�λ����
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
                // ���VIO�в�۲�������֡VIO����֮�����Եġ��������GPS�Ǿ��Ե�
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
                    // ��ȡ��Ա任����iTj����ת���֣��õ������ת��Ԫ��iQj
                    iQj = iTj.block<3, 3>(0, 0);
                    // ��ȡ��Ա任����iTj��ƽ�Ʋ��֣��õ����ƽ������iPj
                    Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);
                    // ����һ��Ceres�Ż���CostFunction����������������ڼ�������˶����Ĵ��ۺ��������д��������ƽ�ƺ������ת�Լ���Ӧ��Ȩ�ء�
                    ceres::CostFunction *vio_function = RelativeRTError::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                                iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                                0.1, 0.01);
                    // �����������Ĵ��ۺ�����ӵ�Ceres�Ż���������Ϊ�в�顣����ʹ������Ԫ����ƽ��������Ϊ�Ż�����
                    // NULL��ʾʹ��Ĭ�ϵ�ƽ����ʧ����������Ϳ������Ҫʹ���Ż�֮���vio�����ݸ���ƽ�������Ǿ��ȣ���Ϊֻ�е�ÿһ�ε�
                    // ��������ȵ�ʱ�򣬲Ż�ﵽ��С��ƽ����
                    // vio_function�ڴ���ʱʹ����localPoseMap��������֡����ƽ��������Ԫ����Ȼ����Ӳв���ʱ��ʹ��globalPoseMap��
                    // ���ڵ�t_array��q_array����ƽ��������Ԫ���������������Ӧ�����ֱ�������������Ĳв
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
                double t = iterVIO->first; // t��ʾ����ʱ��
                // ��GPSPositionMap�в����Ƿ����뵱ǰʱ�����ƥ���GPS���ݡ�GPSPositionMap��һ��ӳ�䣬����ʱ���ӳ�䵽��Ӧ��GPSλ����Ϣ
                // GPS�в����۲���ֱ�Ӿ���GPS�Ĳ������ݣ�
                // �в�������GPS���Ż������Ĳ����Ǿ��ԵĲ
                iterGPS = GPSPositionMap.find(t);

                if (iterGPS != GPSPositionMap.end())
                {
                    // �����ҵ���GPS���ݴ���һ��Ceres�Ż���CostFunction��������������ڼ���VIO���ƵĹ켣��GPS�켣֮�����
                    // GPSλ����Ϣͨ���ɾ��ȡ�γ�ȡ��߶Ⱥ�Ȩ�أ���Э�����ɣ�����ʹ������Щ��Ϣ������һ�����ۺ���
                    ceres::CostFunction *gps_function = TError::Create(iterGPS->second[0], iterGPS->second[1],
                                                                       iterGPS->second[2], iterGPS->second[3]);
                    // printf("inverse weight %f \n", iterGPS->second[3]);
                    // �����������Ĵ��ۺ�����ӵ�Ceres�Ż���������Ϊ�в�顣����в���������Ż�VIO�켣��
                    // ����VIO���ƵĹ켣��GPS�켣���бȽϣ���ͨ����С������֮��Ĳ���������VIO���ƵĽ����t_array[i]��VIO�Ż�����������ʾʱ�����Ӧ��λ�ˡ�
                    // Ceres���������ݴ��ۺ�������ʧ�������������t_array�������е�ֵ����С���в

                    // ����gps_function���������ʹ��t_array[i]����GPS�켣�Ĳвgps_function��һ���º������ڴ�������ʱ��
                    // ʹ����iterGPS->second[0], iterGPS->second[1], iterGPS->second[2], iterGPS->second[3]�������ˣ����൱��
                    // ���ڲ�ά��������ֵ�����ֵ������Ҫ��t_array����ȥ�ģ����γ���vio��GPS�Ĳв��Ϊt_array��������globalPoseMap�ģ�
                    // globalPoseMap��δ���Ż���vio�����ݣ���ô�������ĺ������Ҫ��С����Ӧ��GPS��vio��λ�õľ��룬
                    // loss_function��һ��Huber��ʧ���������ڵ����������Ĵ�С����ʼ��ʱ���õ���1.0��������С�ڵ���1.0����ʹ��Ĭ�ϵ�ƽ����
                    // �����ʧ������1.0����ʹ��Huber��ʧ��������Ե���û����ƽ����Ч�������൱�ڽ�������ʧ�Ĺ��׳̶ȣ����Ż������У����쳣ֵ��
                    // ������Եø����ºͣ��������쳣ֵ��������Ӱ�졣

                    // ʹ�� HuberLoss �ľ��������ǣ�
                    // ʹ�Ż����̶��쳣ֵ��outliers�����и��õ�³���ԡ�
                    // ��С�в����ƽ���������׼L2������ͬ�����Դ�в�������Դ����Ӷ���С�쳣ֵ��������Ӱ�졣
                    // ����Ż�������Ƚ��ԣ�ʹ�����ս�������ױ������쳣ֵ������Ӱ�졣
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
            // ����������Ż�֮��t_array��q_array�е����ݾ����Ż����ȫ��λ��
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++)
            {
                // ��VIO�Ż���λ����������ȡλ�ú���Ԫ��������һ������ȫ��λ�˵�����globalPose
                vector<double> globalPose{t_array[i][0], t_array[i][1], t_array[i][2],
                                          q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
                // ����ȫ��λ��ӳ���е�ǰʱ�����Ӧ��ֵΪ�µ�ȫ��λ��
                iter->second = globalPose;
                // ����Ѿ����������һ��λ������
                // ����VIO����ϵ�µ�λ�ã�localPoseMap����Ż�����GPS����ϵ�µ�λ�ã�globalPose�������WGPS_T_WVIO���и���
                if (i == length - 1)
                {
                    Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                    // ��ȡ��ǰʱ���
                    double t = iter->first;
                    // ��localPoseMap��ȡVIO�������ϵ����������ϵ����ת����
                    WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4],
                                                                       localPoseMap[t][5], localPoseMap[t][6])
                                                        .toRotationMatrix();
                    // ��localPoseMap��ȡVIO�������ϵ����������ϵ��ƽ������
                    WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                    // ��globalPose��ȡGPS����������ϵ����ת����
                    WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4],
                                                                       globalPose[5], globalPose[6])
                                                        .toRotationMatrix();
                    // ��globalPose��ȡGPS����������ϵ��ƽ������
                    WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                    // ����GPS��VIO�ı任����WGPS_T_WVIO
                    // ���WGPS_T_WVIO֮ǰ�ǵ�λ���󣬶���һ�θ��£������������ʵ����������
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

// ���global_path.poses�����¸���globalPoseMap����
void GlobalOptimization::updateGlobalPath()
{
    // ���ȫ��·���е�����λ��
    global_path.poses.clear();
    // ����һ��������iter�����ڱ���globalPoseMap
    map<double, vector<double>>::iterator iter;
    // �Ż�֮��globalPoseMap�������������Ż�������ݣ�������Щ���Ż��˵�t_array, q_array����
    for (iter = globalPoseMap.begin(); iter != globalPoseMap.end(); iter++)
    {
        // ����һ��geometry_msgs::PoseStamped���͵ı���pose_stamped�����ڴ洢λ����Ϣ
        geometry_msgs::PoseStamped pose_stamped;
        // ����pose_stamped��ʱ�����ʹ��globalPoseMap�е�ʱ�����������iter�ļ���
        pose_stamped.header.stamp = ros::Time(iter->first);
        // ����pose_stamped������ϵΪ��world��
        pose_stamped.header.frame_id = "world";
        // ����pose_stamped��λ����Ϣ��ʹ��globalPoseMap�е�x���꣨������iter��ֵ�ĵ�һ��Ԫ�أ�
        pose_stamped.pose.position.x = iter->second[0];
        pose_stamped.pose.position.y = iter->second[1];
        pose_stamped.pose.position.z = iter->second[2];
        pose_stamped.pose.orientation.w = iter->second[3];
        pose_stamped.pose.orientation.x = iter->second[4];
        pose_stamped.pose.orientation.y = iter->second[5];
        pose_stamped.pose.orientation.z = iter->second[6];
        // �������õ�pose_stamped��ӵ�global_path��poses������
        global_path.poses.push_back(pose_stamped);
    }
}