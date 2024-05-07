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

#include "solve_5pts.h"

namespace cv
{
    void decomposeEssentialMat(InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t)
    {

        Mat E = _E.getMat().reshape(1, 3);
        CV_Assert(E.cols == 3 && E.rows == 3);

        Mat D, U, Vt;
        SVD::compute(E, D, U, Vt);

        if (determinant(U) < 0)
            U *= -1.;
        if (determinant(Vt) < 0)
            Vt *= -1.;

        Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        W.convertTo(W, E.type());

        Mat R1, R2, t;
        R1 = U * W * Vt;
        R2 = U * W.t() * Vt;
        t = U.col(2) * 1.0;

        R1.copyTo(_R1);
        R2.copyTo(_R2);
        t.copyTo(_t);
    }

    int recoverPose(InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                    OutputArray _R, OutputArray _t, InputOutputArray _mask)
    {

        Mat points1, points2, cameraMatrix;
        _points1.getMat().convertTo(points1, CV_64F);
        _points2.getMat().convertTo(points2, CV_64F);
        _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

        int npoints = points1.checkVector(2);
        CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
                  points1.type() == points2.type());

        CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

        if (points1.channels() > 1)
        {
            points1 = points1.reshape(1, npoints);
            points2 = points2.reshape(1, npoints);
        }

        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);

        points1.col(0) = (points1.col(0) - cx) / fx;
        points2.col(0) = (points2.col(0) - cx) / fx;
        points1.col(1) = (points1.col(1) - cy) / fy;
        points2.col(1) = (points2.col(1) - cy) / fy;

        points1 = points1.t();
        points2 = points2.t();

        Mat R1, R2, t;
        decomposeEssentialMat(E, R1, R2, t);
        Mat P0 = Mat::eye(3, 4, R1.type());
        Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
        P1(Range::all(), Range(0, 3)) = R1 * 1.0;
        P1.col(3) = t * 1.0;
        P2(Range::all(), Range(0, 3)) = R2 * 1.0;
        P2.col(3) = t * 1.0;
        P3(Range::all(), Range(0, 3)) = R1 * 1.0;
        P3.col(3) = -t * 1.0;
        P4(Range::all(), Range(0, 3)) = R2 * 1.0;
        P4.col(3) = -t * 1.0;

        // Do the cheirality check.
        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between postive and negtive.
        double dist = 50.0;
        Mat Q;
        triangulatePoints(P0, P1, points1, points2, Q);
        Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask1 = (Q.row(2) < dist) & mask1;
        Q = P1 * Q;
        mask1 = (Q.row(2) > 0) & mask1;
        mask1 = (Q.row(2) < dist) & mask1;

        triangulatePoints(P0, P2, points1, points2, Q);
        Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask2 = (Q.row(2) < dist) & mask2;
        Q = P2 * Q;
        mask2 = (Q.row(2) > 0) & mask2;
        mask2 = (Q.row(2) < dist) & mask2;

        triangulatePoints(P0, P3, points1, points2, Q);
        Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask3 = (Q.row(2) < dist) & mask3;
        Q = P3 * Q;
        mask3 = (Q.row(2) > 0) & mask3;
        mask3 = (Q.row(2) < dist) & mask3;

        triangulatePoints(P0, P4, points1, points2, Q);
        Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask4 = (Q.row(2) < dist) & mask4;
        Q = P4 * Q;
        mask4 = (Q.row(2) > 0) & mask4;
        mask4 = (Q.row(2) < dist) & mask4;

        mask1 = mask1.t();
        mask2 = mask2.t();
        mask3 = mask3.t();
        mask4 = mask4.t();

        // If _mask is given, then use it to filter outliers.
        if (!_mask.empty())
        {
            Mat mask = _mask.getMat();
            CV_Assert(mask.size() == mask1.size());
            bitwise_and(mask, mask1, mask1);
            bitwise_and(mask, mask2, mask2);
            bitwise_and(mask, mask3, mask3);
            bitwise_and(mask, mask4, mask4);
        }
        if (_mask.empty() && _mask.needed())
        {
            _mask.create(mask1.size(), CV_8U);
        }

        CV_Assert(_R.needed() && _t.needed());
        _R.create(3, 3, R1.type());
        _t.create(3, 1, t.type());

        int good1 = countNonZero(mask1);
        int good2 = countNonZero(mask2);
        int good3 = countNonZero(mask3);
        int good4 = countNonZero(mask4);

        if (good1 >= good2 && good1 >= good3 && good1 >= good4)
        {
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed())
                mask1.copyTo(_mask);
            return good1;
        }
        else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
        {
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed())
                mask2.copyTo(_mask);
            return good2;
        }
        else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
        {
            t = -t;
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed())
                mask3.copyTo(_mask);
            return good3;
        }
        else
        {
            t = -t;
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed())
                mask4.copyTo(_mask);
            return good4;
        }
    }

    int recoverPose(InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                    OutputArray _t, double focal, Point2d pp, InputOutputArray _mask)
    {
        Mat cameraMatrix = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
        return cv::recoverPose(E, _points1, _points2, cameraMatrix, _R, _t, _mask);
    }
}

// solveRelativeRT ()利用对积几何的知识，通过分解本质矩阵求得：从最后一帧到参考帧，发生的位姿变化（旋转Rotation和位移Translation），存入relative_R 和relative_T 。
bool MotionEstimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    // 检查对应关系向量的大小是否大于或等于15，以确保有足够的对应关系用于计算相对运动
    if (corres.size() >= 15) // 这个条件一直成立
    {
        // 将对应关系向量中的点对转换为 cv::Point2f 类型的向量 ll 和 rr，用于 OpenCV 中的计算
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        // 使用 RANSAC 方法计算基础矩阵 E，用于估计相机之间的本质矩阵。mask 是用于存储内点的标志向量。
        // findFundamentalMat() 是通过归一化坐标求得的本质矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        // 通过基础矩阵恢复相对位姿，其中 rot 和 trans 是相对旋转和平移的输出。inlier_cnt 是内点的数量
        // 这里cv::recoverPose()函数不是OpenCV源代码的那个recoverPose()，而是自己实现的，在solve_5pts.cpp中，通过本质矩阵恢复出
        // 旋转和平移信息，并得到内点数目，即认为在对极几何中起作用的特征点（踢除了深度为负或者深度过大的点)
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        // cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        // 将 OpenCV 中的旋转矩阵和平移向量转换为 Eigen 库中的矩阵和向量
        for (int i = 0; i < 3; i++)
        {
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }
        // 将旋转矩阵转置后赋值给 Rotation，并计算平移向量并赋值给 Translation
        // 最终获取从最后一帧到参考帧，发生的位姿变化（旋转Rotation和位移Translation）
        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        // 码检查内点的数量是否大于12，如果是，则返回 true，否则返回 false
        if (inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}
