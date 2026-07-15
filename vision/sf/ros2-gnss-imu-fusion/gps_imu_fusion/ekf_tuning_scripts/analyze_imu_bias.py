#!/usr/bin/env python3
# analyze_imu_bias.py
# 정적 데이터에서 IMU 센서의 바이어스 안정성을 분석하는 스크립트

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path
from datetime import datetime
import argparse
import time

class IMUBiasAnalyzer(Node):
    def __init__(self, duration=300.0, output_dir='ekf_param_results'):
        """IMU 바이어스 안정성을 분석하는 노드
        
        Args:
            duration: 데이터 수집 시간(초), 바이어스 안정성은 5분 이상 권장
            output_dir: 결과 저장 디렉토리
        """
        super().__init__('imu_bias_analyzer')
        
        # 파라미터 설정
        self.duration = duration
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장용 배열
        self.accel_data = []
        self.gyro_data = []
        self.timestamps = []
        
        # 데이터 수집 시작 시간
        self.start_time = time.time()
        
        # IMU 토픽 구독
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10  # QoS 설정
        )
        
        # 타이머 설정 (지정된 시간 후 분석 시작)
        self.timer = self.create_timer(self.duration, self.analyze_data)
        
        self.get_logger().info(f"IMU 바이어스 안정성 분석 시작 - {self.duration}초 동안 데이터 수집")
        self.get_logger().info("주의: 바이어스 안정성 분석을 위해 IMU가 움직이지 않아야 합니다.")
    
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
        
        # 시간은 테스트 시작부터의 경과 시간(초)으로 저장
        time_now = time.time() - self.start_time
        
        self.accel_data.append(accel)
        self.gyro_data.append(gyro)
        self.timestamps.append(time_now)
    
    def analyze_data(self):
        """수집된 IMU 데이터 분석"""
        self.get_logger().info(f"데이터 수집 완료. 바이어스 안정성 분석 시작...")
        
        # 타이머 중지 (한 번만 분석)
        self.timer.cancel()
        
        # 충분한 데이터가 있는지 확인
        if len(self.accel_data) < 100:
            self.get_logger().error("충분한 IMU 데이터가 수집되지 않았습니다.")
            return
        
        # 데이터를 numpy 배열로 변환
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        time_array = np.array(self.timestamps)
        
        # 시간 간격 계산 (샘플링 주기)
        dt = np.mean(np.diff(time_array))
        sample_rate = 1 / dt if dt > 0 else 0
        
        # 가속도 바이어스 분석
        accel_bias_result = self.analyze_bias_stability(accel_array, time_array, "가속도계", "m/s²")
        
        # 자이로 바이어스 분석
        gyro_bias_result = self.analyze_bias_stability(gyro_array, time_array, "자이로스코프", "rad/s")
        
        # 추천 EKF 파라미터 계산
        recommended_params = self.calculate_recommended_params(accel_bias_result, gyro_bias_result)
        
        # 결과 저장
        result = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_count': len(self.accel_data),
            'duration_seconds': float(time_array[-1]),
            'sample_rate_hz': float(sample_rate),
            'accel_bias_analysis': accel_bias_result,
            'gyro_bias_analysis': gyro_bias_result,
            'recommended_params': recommended_params
        }
        
        # YAML 파일로 저장
        yaml_file = Path(self.output_dir) / 'imu_bias_analysis.yaml'
        with open(str(yaml_file), 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        self.get_logger().info(f"분석 결과가 {yaml_file}에 저장되었습니다.")
        self.get_logger().info("추천 EKF 파라미터:")
        for key, value in recommended_params.items():
            self.get_logger().info(f"  {key}: {value}")
        
        # 노드 종료
        self.get_logger().info("분석 완료. 노드를 종료합니다.")
        rclpy.shutdown()
    
    def analyze_bias_stability(self, data_array, time_array, sensor_name, unit):
        """센서 바이어스 안정성 분석
        
        Args:
            data_array: 센서 데이터 배열 (N x 3)
            time_array: 타임스탬프 배열 (N)
            sensor_name: 센서 이름 (로그 출력용)
            unit: 측정 단위 (로그 출력용)
        
        Returns:
            dict: 바이어스 안정성 분석 결과
        """
        # 각 축의 평균값을 바이어스로 간주
        bias_mean = np.mean(data_array, axis=0)
        
        # 전체 데이터 기간 동안의 각 축 표준편차
        bias_std = np.std(data_array, axis=0)
        
        # 윈도우 크기에 따른 앨런 분산 분석 (간소화된 버전)
        window_sizes = np.logspace(0, np.log10(len(time_array) // 10), 20).astype(int)
        window_sizes = np.unique(window_sizes)
        
        allan_variance_x = []
        allan_variance_y = []
        allan_variance_z = []
        
        for window in window_sizes:
            if window < 2:  # 윈도우 크기는 최소 2 이상이어야 함
                continue
                
            # 윈도우 별 평균값 계산
            num_windows = len(data_array) // window
            if num_windows < 2:
                break
                
            window_means_x = np.array([np.mean(data_array[i*window:(i+1)*window, 0]) for i in range(num_windows)])
            window_means_y = np.array([np.mean(data_array[i*window:(i+1)*window, 1]) for i in range(num_windows)])
            window_means_z = np.array([np.mean(data_array[i*window:(i+1)*window, 2]) for i in range(num_windows)])
            
            # 인접한 평균들의 차이 제곱의 평균이 앨런 분산
            diff_x = np.diff(window_means_x)
            diff_y = np.diff(window_means_y)
            diff_z = np.diff(window_means_z)
            
            av_x = np.sum(diff_x**2) / (2 * (num_windows - 1))
            av_y = np.sum(diff_y**2) / (2 * (num_windows - 1))
            av_z = np.sum(diff_z**2) / (2 * (num_windows - 1))
            
            allan_variance_x.append(av_x)
            allan_variance_y.append(av_y)
            allan_variance_z.append(av_z)
        
        # 윈도우 시간 계산 (초)
        window_times = window_sizes * np.mean(np.diff(time_array))
        
        # 앨런 분산 그래프 그리기
        self.plot_allan_variance(window_times, allan_variance_x, allan_variance_y, allan_variance_z, 
                                sensor_name, unit)
        
        # 바이어스 안정성 (노이즈와 램덤워크로 모델링)
        # 최소 앨런 편차에서의 바이어스 안정성 (노이즈 요소)
        min_idx_x = np.argmin(allan_variance_x)
        min_idx_y = np.argmin(allan_variance_y)
        min_idx_z = np.argmin(allan_variance_z)
        
        bias_stability_x = np.sqrt(allan_variance_x[min_idx_x])
        bias_stability_y = np.sqrt(allan_variance_y[min_idx_y])
        bias_stability_z = np.sqrt(allan_variance_z[min_idx_z])
        
        bias_stability_time_x = window_times[min_idx_x]
        bias_stability_time_y = window_times[min_idx_y]
        bias_stability_time_z = window_times[min_idx_z]
        
        # 바이어스 변화율 추정 (시간 경과에 따른 경향성)
        # 선형 회귀를 사용하여 단위 시간 당 바이어스 변화율 추정
        slope_x, _, _, _, _ = stats.linregress(time_array, data_array[:, 0])
        slope_y, _, _, _, _ = stats.linregress(time_array, data_array[:, 1])
        slope_z, _, _, _, _ = stats.linregress(time_array, data_array[:, 2])
        
        # 바이어스 시계열 그래프 그리기
        self.plot_bias_timeseries(time_array, data_array, sensor_name, unit)
        
        # 결과 로그 출력
        self.get_logger().info(f"{sensor_name} 바이어스 분석 결과:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.get_logger().info(f"  {axis} 축: 평균 바이어스 = {bias_mean[i]:.6f} {unit}, 표준편차 = {bias_std[i]:.6f} {unit}")
        
        self.get_logger().info(f"  바이어스 안정성:")
        self.get_logger().info(f"    X 축: {bias_stability_x:.6f} {unit} (시간상수: {bias_stability_time_x:.1f}초)")
        self.get_logger().info(f"    Y 축: {bias_stability_y:.6f} {unit} (시간상수: {bias_stability_time_y:.1f}초)")
        self.get_logger().info(f"    Z 축: {bias_stability_z:.6f} {unit} (시간상수: {bias_stability_time_z:.1f}초)")
        
        self.get_logger().info(f"  바이어스 변화율 (드리프트):")
        self.get_logger().info(f"    X 축: {slope_x:.9f} {unit}/s")
        self.get_logger().info(f"    Y 축: {slope_y:.9f} {unit}/s")
        self.get_logger().info(f"    Z 축: {slope_z:.9f} {unit}/s")
        
        # 결과 반환
        return {
            'bias_mean': {
                'x': float(bias_mean[0]), 
                'y': float(bias_mean[1]), 
                'z': float(bias_mean[2])
            },
            'bias_std': {
                'x': float(bias_std[0]), 
                'y': float(bias_std[1]), 
                'z': float(bias_std[2])
            },
            'bias_stability': {
                'x': float(bias_stability_x),
                'y': float(bias_stability_y),
                'z': float(bias_stability_z)
            },
            'bias_stability_time': {
                'x': float(bias_stability_time_x),
                'y': float(bias_stability_time_y),
                'z': float(bias_stability_time_z)
            },
            'bias_drift_rate': {
                'x': float(slope_x),
                'y': float(slope_y),
                'z': float(slope_z)
            }
        }
    
    def calculate_recommended_params(self, accel_bias_result, gyro_bias_result):
        """분석 결과를 바탕으로 추천 파라미터 계산"""
        # 바이어스 노이즈 추정 (최대 바이어스 안정성 값의 비율로 설정)
        accel_max_stability = max(
            accel_bias_result['bias_stability']['x'],
            accel_bias_result['bias_stability']['y'],
            accel_bias_result['bias_stability']['z']
        )
        
        gyro_max_stability = max(
            gyro_bias_result['bias_stability']['x'],
            gyro_bias_result['bias_stability']['y'],
            gyro_bias_result['bias_stability']['z']
        )
        
        # 바이어스 시간 상수 추정 (각 축의 안정성 시간 중 중간값 사용)
        accel_tau_values = [
            accel_bias_result['bias_stability_time']['x'],
            accel_bias_result['bias_stability_time']['y'],
            accel_bias_result['bias_stability_time']['z']
        ]
        
        gyro_tau_values = [
            gyro_bias_result['bias_stability_time']['x'],
            gyro_bias_result['bias_stability_time']['y'],
            gyro_bias_result['bias_stability_time']['z']
        ]
        
        # 중간값 계산
        accel_bias_tau = float(np.median(accel_tau_values))
        gyro_bias_tau = float(np.median(gyro_tau_values))
        
        # 바이어스 노이즈는 바이어스 안정성의 일정 비율로 설정
        accel_bias_noise = float(accel_max_stability / 5)  # 보수적인 추정
        gyro_bias_noise = float(gyro_max_stability / 5)    # 보수적인 추정
        
        # 시간 상수는 측정 시간이 너무 짧으면 기본값 사용
        if accel_bias_tau < 10:
            accel_bias_tau = 100.0  # 기본값
        
        if gyro_bias_tau < 10:
            gyro_bias_tau = 50.0  # 기본값
        
        # 추천 EKF 파라미터 반환
        return {
            'accel_bias_noise': accel_bias_noise,
            'gyro_bias_noise': gyro_bias_noise,
            'accel_bias_tau': accel_bias_tau,
            'gyro_bias_tau': gyro_bias_tau
        }
    
    def plot_allan_variance(self, window_times, av_x, av_y, av_z, sensor_name, unit):
        """앨런 분산 그래프 그리기"""
        plt.figure(figsize=(10, 6))
        
        plt.loglog(window_times, np.sqrt(av_x), 'r-', label='X 축')
        plt.loglog(window_times, np.sqrt(av_y), 'g-', label='Y 축')
        plt.loglog(window_times, np.sqrt(av_z), 'b-', label='Z 축')
        
        plt.title(f'{sensor_name} 앨런 편차')
        plt.xlabel('평균 시간 (초)')
        plt.ylabel(f'앨런 편차 ({unit})')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        
        # 그래프 저장
        plt_file = Path(self.output_dir) / f'{sensor_name.lower()}_allan_variance.png'
        plt.savefig(str(plt_file))
        plt.close()
        
        self.get_logger().info(f"{sensor_name} 앨런 분산 그래프가 {plt_file}에 저장되었습니다.")
    
    def plot_bias_timeseries(self, time_array, data_array, sensor_name, unit):
        """바이어스 시계열 그래프 그리기"""
        plt.figure(figsize=(12, 8))
        
        plt.plot(time_array, data_array[:, 0], 'r-', label='X 축')
        plt.plot(time_array, data_array[:, 1], 'g-', label='Y 축')
        plt.plot(time_array, data_array[:, 2], 'b-', label='Z 축')
        
        plt.title(f'{sensor_name} 바이어스 시계열')
        plt.xlabel('시간 (초)')
        plt.ylabel(f'{unit}')
        plt.grid(True)
        plt.legend()
        
        # 그래프 저장
        plt_file = Path(self.output_dir) / f'{sensor_name.lower()}_bias_timeseries.png'
        plt.savefig(str(plt_file))
        plt.close()
        
        self.get_logger().info(f"{sensor_name} 바이어스 시계열 그래프가 {plt_file}에 저장되었습니다.")


def main(args=None):
    parser = argparse.ArgumentParser(description='IMU 바이어스 안정성 분석 도구')
    parser.add_argument('-d', '--duration', type=float, default=300.0,
                        help='데이터 수집 기간(초), 기본값: 300초 (5분)')
    parser.add_argument('-o', '--output-dir', type=str, default='ekf_param_results',
                        help='분석 결과 저장 디렉토리, 기본값: ekf_param_results')
    
    parsed_args = parser.parse_args(args=args)
    
    print(f"IMU 바이어스 안정성 분석 시작:")
    print(f"- 데이터 수집 기간: {parsed_args.duration}초")
    print(f"- 결과 저장 위치: {parsed_args.output_dir}")
    print("주의: 바이어스 안정성 분석을 위해 IMU가 움직이지 않아야 합니다.")
    
    rclpy.init(args=args)
    node = IMUBiasAnalyzer(
        duration=parsed_args.duration,
        output_dir=parsed_args.output_dir
    )
    rclpy.spin(node)
    
    # spin 종료 후 필요한 정리 작업
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
