#!/usr/bin/env python3
# analyze_imu_noise.py
# 정적 데이터에서 IMU 센서의 노이즈 특성을 분석하는 스크립트

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import argparse

class IMUNoiseAnalyzer(Node):
    def __init__(self, duration=30.0, output_dir='ekf_param_results'):
        """IMU 노이즈를 분석하는 노드
        
        Args:
            duration: 데이터 수집 시간(초)
            output_dir: 결과 저장 디렉토리
        """
        super().__init__('imu_noise_analyzer')
        
        # 파라미터 설정
        self.duration = duration
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장용 배열
        self.accel_data = []
        self.gyro_data = []
        self.timestamps = []
        
        # IMU 토픽 구독
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10  # QoS 설정
        )
        
        # 타이머 설정 (지정된 시간 후 분석 시작)
        self.timer = self.create_timer(self.duration, self.analyze_data)
        
        self.get_logger().info(f"IMU 노이즈 분석 시작 - {self.duration}초 동안 데이터 수집")
    
    def imu_callback(self, msg):
        """IMU 메시지 콜백 함수"""
        # IMU 데이터 저장
        accel = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]
        
        gyro = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        
        time = self.get_clock().now().nanoseconds / 1e9  # 초 단위로 변환
        
        self.accel_data.append(accel)
        self.gyro_data.append(gyro)
        self.timestamps.append(time)
    
    def analyze_data(self):
        """수집된 IMU 데이터 분석"""
        self.get_logger().info(f"데이터 수집 완료. 분석 시작...")
        
        # 타이머 중지 (한 번만 분석)
        self.timer.cancel()
        
        # 충분한 데이터가 있는지 확인
        if len(self.accel_data) < 10:
            self.get_logger().error("충분한 IMU 데이터가 수집되지 않았습니다.")
            return
        
        # 데이터를 numpy 배열로 변환
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        time_array = np.array(self.timestamps)
        
        # 시간 간격 계산 (샘플링 주기)
        dt = np.mean(np.diff(time_array))
        sample_rate = 1 / dt if dt > 0 else 0
        
        # 각 축별 평균과 표준편차 계산
        accel_means = np.mean(accel_array, axis=0)
        accel_stds = np.std(accel_array, axis=0)
        
        gyro_means = np.mean(gyro_array, axis=0)
        gyro_stds = np.std(gyro_array, axis=0)
        
        # 결과 출력
        self.get_logger().info(f"샘플 수: {len(self.accel_data)}, 샘플링 주기: {dt:.6f}s, 샘플링 레이트: {sample_rate:.1f}Hz")
        
        self.get_logger().info("가속도계 노이즈 분석:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.get_logger().info(f"  {axis} 축: 평균 = {accel_means[i]:.6f} m/s², 표준편차 = {accel_stds[i]:.6f} m/s²")
        
        self.get_logger().info("자이로스코프 노이즈 분석:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.get_logger().info(f"  {axis} 축: 평균 = {gyro_means[i]:.6f} rad/s, 표준편차 = {gyro_stds[i]:.6f} rad/s")
        
        # 추천 EKF 파라미터 계산
        accel_noise = float(np.max(accel_stds))  # 최대 표준편차 사용
        gyro_noise = float(np.max(gyro_stds))    # 최대 표준편차 사용
        
        # 가속도계와 자이로 바이어스 노이즈는 노이즈의 일부로 추정
        accel_bias_noise = float(accel_noise * 0.2)  # 20%로 추정
        gyro_bias_noise = float(gyro_noise * 0.1)    # 10%로 추정
        
        # 그래프 그리기
        self.plot_imu_data(accel_array, gyro_array, time_array)
        
        # 결과 저장
        result = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_count': len(self.accel_data),
            'sample_rate_hz': float(sample_rate),
            'accel_stats': {
                'x': {'mean': float(accel_means[0]), 'std': float(accel_stds[0])},
                'y': {'mean': float(accel_means[1]), 'std': float(accel_stds[1])},
                'z': {'mean': float(accel_means[2]), 'std': float(accel_stds[2])}
            },
            'gyro_stats': {
                'x': {'mean': float(gyro_means[0]), 'std': float(gyro_stds[0])},
                'y': {'mean': float(gyro_means[1]), 'std': float(gyro_stds[1])},
                'z': {'mean': float(gyro_means[2]), 'std': float(gyro_stds[2])}
            },
            'recommended_params': {
                'accel_noise': accel_noise,
                'gyro_noise': gyro_noise,
                'accel_bias_noise': accel_bias_noise,
                'gyro_bias_noise': gyro_bias_noise
            }
        }
        
        # YAML 파일로 저장
        yaml_file = Path(self.output_dir) / 'imu_noise_analysis.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        self.get_logger().info(f"분석 결과가 {yaml_file}에 저장되었습니다.")
        self.get_logger().info("추천 EKF 파라미터:")
        self.get_logger().info(f"  accel_noise: {accel_noise:.6f}")
        self.get_logger().info(f"  gyro_noise: {gyro_noise:.6f}")
        self.get_logger().info(f"  accel_bias_noise: {accel_bias_noise:.6f}")
        self.get_logger().info(f"  gyro_bias_noise: {gyro_bias_noise:.6f}")
        
        # 노드 종료
        self.get_logger().info("분석 완료. 노드를 종료합니다.")
        rclpy.shutdown()
    
    def plot_imu_data(self, accel_data, gyro_data, time_data):
        """IMU 데이터를 시각화하여 PNG 파일로 저장"""
        # 상대 시간으로 변환 (첫 타임스탬프를 0으로)
        rel_time = time_data - time_data[0]
        
        plt.figure(figsize=(12, 8))
        
        # 가속도 그래프
        plt.subplot(2, 1, 1)
        plt.plot(rel_time, accel_data[:, 0], 'r-', label='X')
        plt.plot(rel_time, accel_data[:, 1], 'g-', label='Y')
        plt.plot(rel_time, accel_data[:, 2], 'b-', label='Z')
        plt.title('가속도계 데이터')
        plt.xlabel('시간 (초)')
        plt.ylabel('가속도 (m/s²)')
        plt.grid(True)
        plt.legend()
        
        # 각속도 그래프
        plt.subplot(2, 1, 2)
        plt.plot(rel_time, gyro_data[:, 0], 'r-', label='X')
        plt.plot(rel_time, gyro_data[:, 1], 'g-', label='Y')
        plt.plot(rel_time, gyro_data[:, 2], 'b-', label='Z')
        plt.title('자이로스코프 데이터')
        plt.xlabel('시간 (초)')
        plt.ylabel('각속도 (rad/s)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 그래프 저장
        plt_file = Path(self.output_dir) / 'imu_noise_analysis.png'
        plt.savefig(str(plt_file))
        plt.close()
        
        self.get_logger().info(f"IMU 데이터 그래프가 {plt_file}에 저장되었습니다.")


def main(args=None):
    parser = argparse.ArgumentParser(description='IMU 노이즈 분석 도구')
    parser.add_argument('-d', '--duration', type=float, default=30.0,
                        help='데이터 수집 기간(초), 기본값: 30초')
    parser.add_argument('-o', '--output-dir', type=str, default='ekf_param_results',
                        help='분석 결과 저장 디렉토리, 기본값: ekf_param_results')
    
    parsed_args = parser.parse_args(args=args)
    
    print(f"IMU 노이즈 분석 시작:")
    print(f"- 데이터 수집 기간: {parsed_args.duration}초")
    print(f"- 결과 저장 위치: {parsed_args.output_dir}")
    
    rclpy.init(args=args)
    node = IMUNoiseAnalyzer(
        duration=parsed_args.duration,
        output_dir=parsed_args.output_dir
    )
    rclpy.spin(node)
    
    # spin 종료 후 필요한 정리 작업
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
