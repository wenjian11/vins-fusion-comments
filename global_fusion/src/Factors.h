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

#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// 四元素的逆就是xyz取相反数
template <typename T>
inline void QuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[0] = q[0];
	q_inverse[1] = -q[1];
	q_inverse[2] = -q[2];
	q_inverse[3] = -q[3];
};

struct TError
{
	TError(double t_x, double t_y, double t_z, double var)
		: t_x(t_x), t_y(t_y), t_z(t_z), var(var) {}
	// 首先是GPS的数据和状态量定义的残差：状态量位置-gps算出来的位置
	template <typename T>
	bool operator()(const T *tj, T *residuals) const
	{
		residuals[0] = (tj[0] - T(t_x)) / T(var);
		residuals[1] = (tj[1] - T(t_y)) / T(var);
		residuals[2] = (tj[2] - T(t_z)) / T(var);

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z, const double var)
	{
		return (new ceres::AutoDiffCostFunction<
				TError, 3, 3>(
			new TError(t_x, t_y, t_z, var)));
	}
	// double t_x, t_y, t_z;: GPS数据的x、y、z坐标。
	// var : 方差，用于归一化残差。
	double t_x, t_y, t_z, var;
};

// 该结构体用于将一个帧相对于另一个帧的位姿差异转换为残差
struct RelativeRTError
{
	RelativeRTError(double t_x, double t_y, double t_z,
					double q_w, double q_x, double q_y, double q_z,
					double t_var, double q_var)
		: t_x(t_x), t_y(t_y), t_z(t_z),
		  q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		  t_var(t_var), q_var(q_var) {}

	template <typename T>
	bool operator()(const T *const w_q_i, const T *ti, const T *w_q_j, const T *tj, T *residuals) const
	{
		// t_w_ij 表示在世界坐标系下，第 i 帧到第 j 帧的位置增量
		T t_w_ij[3]; // 世界坐标系下ij帧的位置增量
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4]; // i帧的四元数逆
		QuaternionInverse(w_q_i, i_q_w);
		// 旋转位置增量到i帧坐标系
		T t_i_ij[3]; // i帧坐标系下，ij帧的位置增量
		// 用于将一个三维点旋转到另一个坐标系中
		// 函数签名为 void QuaternionRotatePoint(const T q[4], const T pt[3], T result[3])：
		// q 是输入的四元数，表示旋转。
		// pt 是输入的三维点（向量）。
		// result 是输出的三维点（向量），表示旋转后的结果。
		// t_w_ij 在世界坐标系下表示的位移向量通过 i_q_w 旋转到参考帧 i 的坐标系中，结果存储在 t_i_ij 中。
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);
		// 计算位置残差并归一化
		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);
		// 计算传入的相对旋转四元数
		T relative_q[4]; // 传入观测的四元数增量
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);
		// 计算状态量的相对旋转四元数
		T q_i_j[4]; // 状态量计算的四元数增量，使用了世界坐标系作为中转
		// i_q_w 是从世界坐标系到参考帧 i 的四元数变换的逆
		// w_q_j 是从世界坐标系到目标帧 j 的四元数变换
		// q_i_j 表示从参考帧 i 到目标帧 j 的四元数变换，通过组合 i_q_w 和 w_q_j 计算得到
		// 具体来说，q_i_j 是 i 帧到世界坐标系的旋转（i_q_w 的逆）和世界坐标系到 j 帧的旋转（w_q_j）的组合
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);
		// 计算传入的相对旋转四元数的逆
		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);
		// 计算旋转残差
		T error_q[4]; // 状态量计算的增量乘上测量量的逆，定义了残差
		// 误差四元数 error_q 是通过状态量计算的四元数和测量量的逆之间的乘积得到的
		// 具体来说，relative_q_inv 是观测值的逆（测量量的逆），q_i_j 是通过状态量计算的值
		// 计算出误差四元数 error_q，表示实际观测值和计算值之间的差异
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);
		// 提取并归一化旋转残差的虚部
		residuals[3] = T(2) * error_q[1] / T(q_var);
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double q_w, const double q_x, const double q_y, const double q_z,
									   const double t_var, const double q_var)
	{
		// 6: 残差的维度，即 residuals 数组的长度为 6。
		// 4: 第一个参数块的维度，即 w_q_i（第一个帧的四元数）的长度为 4。
		// 3: 第二个参数块的维度，即 ti（第一个帧的位置）的长度为 3。
		// 4: 第三个参数块的维度，即 w_q_j（第二个帧的四元数）的长度为 4。
		// 3: 第四个参数块的维度，即 tj（第二个帧的位置）的长度为 3。
		// 6, 4, 3, 4, 3 这五个数字分别表示残差的维度以及每个输入参数块的维度，用于构造自动微分代价函数
		return (new ceres::AutoDiffCostFunction<
				RelativeRTError, 6, 4, 3, 4, 3>(
			new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	double t_x, t_y, t_z, t_norm;
	double q_w, q_x, q_y, q_z;
	double t_var, q_var;
};