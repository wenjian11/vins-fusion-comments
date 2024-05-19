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

// ��Ԫ�ص������xyzȡ�෴��
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
	// ������GPS�����ݺ�״̬������Ĳв״̬��λ��-gps�������λ��
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
	// double t_x, t_y, t_z;: GPS���ݵ�x��y��z���ꡣ
	// var : ������ڹ�һ���в
	double t_x, t_y, t_z, var;
};

// �ýṹ�����ڽ�һ��֡�������һ��֡��λ�˲���ת��Ϊ�в�
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
		// t_w_ij ��ʾ����������ϵ�£��� i ֡���� j ֡��λ������
		T t_w_ij[3]; // ��������ϵ��ij֡��λ������
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4]; // i֡����Ԫ����
		QuaternionInverse(w_q_i, i_q_w);
		// ��תλ��������i֡����ϵ
		T t_i_ij[3]; // i֡����ϵ�£�ij֡��λ������
		// ���ڽ�һ����ά����ת����һ������ϵ��
		// ����ǩ��Ϊ void QuaternionRotatePoint(const T q[4], const T pt[3], T result[3])��
		// q ���������Ԫ������ʾ��ת��
		// pt ���������ά�㣨��������
		// result ���������ά�㣨����������ʾ��ת��Ľ����
		// t_w_ij ����������ϵ�±�ʾ��λ������ͨ�� i_q_w ��ת���ο�֡ i ������ϵ�У�����洢�� t_i_ij �С�
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);
		// ����λ�òв��һ��
		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);
		// ���㴫��������ת��Ԫ��
		T relative_q[4]; // ����۲����Ԫ������
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);
		// ����״̬���������ת��Ԫ��
		T q_i_j[4]; // ״̬���������Ԫ��������ʹ������������ϵ��Ϊ��ת
		// i_q_w �Ǵ���������ϵ���ο�֡ i ����Ԫ���任����
		// w_q_j �Ǵ���������ϵ��Ŀ��֡ j ����Ԫ���任
		// q_i_j ��ʾ�Ӳο�֡ i ��Ŀ��֡ j ����Ԫ���任��ͨ����� i_q_w �� w_q_j ����õ�
		// ������˵��q_i_j �� i ֡����������ϵ����ת��i_q_w ���棩����������ϵ�� j ֡����ת��w_q_j�������
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);
		// ���㴫��������ת��Ԫ������
		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);
		// ������ת�в�
		T error_q[4]; // ״̬��������������ϲ��������棬�����˲в�
		// �����Ԫ�� error_q ��ͨ��״̬���������Ԫ���Ͳ���������֮��ĳ˻��õ���
		// ������˵��relative_q_inv �ǹ۲�ֵ���棨���������棩��q_i_j ��ͨ��״̬�������ֵ
		// ����������Ԫ�� error_q����ʾʵ�ʹ۲�ֵ�ͼ���ֵ֮��Ĳ���
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);
		// ��ȡ����һ����ת�в���鲿
		residuals[3] = T(2) * error_q[1] / T(q_var);
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double q_w, const double q_x, const double q_y, const double q_z,
									   const double t_var, const double q_var)
	{
		// 6: �в��ά�ȣ��� residuals ����ĳ���Ϊ 6��
		// 4: ��һ���������ά�ȣ��� w_q_i����һ��֡����Ԫ�����ĳ���Ϊ 4��
		// 3: �ڶ����������ά�ȣ��� ti����һ��֡��λ�ã��ĳ���Ϊ 3��
		// 4: �������������ά�ȣ��� w_q_j���ڶ���֡����Ԫ�����ĳ���Ϊ 4��
		// 3: ���ĸ��������ά�ȣ��� tj���ڶ���֡��λ�ã��ĳ���Ϊ 3��
		// 6, 4, 3, 4, 3 ��������ֱַ��ʾ�в��ά���Լ�ÿ������������ά�ȣ����ڹ����Զ�΢�ִ��ۺ���
		return (new ceres::AutoDiffCostFunction<
				RelativeRTError, 6, 4, 3, 4, 3>(
			new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	double t_x, t_y, t_z, t_norm;
	double q_w, q_x, q_y, q_z;
	double t_var, q_var;
};