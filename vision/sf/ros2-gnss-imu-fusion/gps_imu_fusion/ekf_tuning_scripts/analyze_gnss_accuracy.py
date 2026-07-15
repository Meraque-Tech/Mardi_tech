#!/usr/bin/env python3
# analyze_gnss_accuracy.py
# 정적 데이터에서 GNSS 센서의 정확도를 분석하는 스크립트

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistWithCovarianceStamped
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import argparse

class GNSSAccuracyAnalyzer(Node):
    def __init__(self, duration=60.0, output_dir='ekf_param_results'):
        """GNSS 정확도를 분석하는 노드
        
        Args:
            duration: 데이터 수집 시간(초)
            output_dir: 결과 저장 디렉토리
        """
        super().__init__('gnss_accuracy_analyzer')
        
        # 파라미터 설정
        self.duration = duration
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장용 배열
        self.fix_data = []  # NavSatFix 메시지 저장
        self.vel_data = []  # TwistWithCovarianceStamped 메시지 저장
        self.fix_timestamps = []
        self.vel_timestamps = []
        
        # GNSS 토픽 구독
        self.fix_subscription = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',
            self.fix_callback,
            10  # QoS 설정
        )
        
        self.vel_subscription = self.create_subscription(
            TwistWithCovarianceStamped,
            '/ublox_gps_node/fix_velocity',
            self.vel_callback,
            10  # QoS 설정
        )
        
        # 타이머 설정 (지정된 시간 후 분석 시작)
        self.timer = self.create_timer(self.duration, self.analyze_data)
        
        self.get_logger().info(f"GNSS 정확도 분석 시작 - {self.duration}초 동안 데이터 수집")
    
    def fix_callback(self, msg):
        """NavSatFix 메시지 콜백 함수"""
        # Fix 데이터 저장
        fix_data = {
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'position_covariance': list(msg.position_covariance),
            'position_covariance_type': msg.position_covariance_type,
            'status': msg.status.status,
            'service': msg.status.service
        }
        
        time = self.get_clock().now().nanoseconds / 1e9  # 초 단위로 변환
        
        self.fix_data.append(fix_data)
        self.fix_timestamps.append(time)
    
    def vel_callback(self, msg):
        """TwistWithCovarianceStamped 메시지 콜백 함수"""
        # 속도 데이터 저장
        vel_data = {
            'linear': {
                'x': msg.twist.twist.linear.x,
                'y': msg.twist.twist.linear.y,
                'z': msg.twist.twist.linear.z
            },
            'angular': {
                'x': msg.twist.twist.angular.x,
                'y': msg.twist.twist.angular.y,
                'z': msg.twist.twist.angular.z
            },
            'covariance': list(msg.twist.covariance)
        }
        
        time = self.get_clock().now().nanoseconds / 1e9  # 초 단위로 변환
        
        self.vel_data.append(vel_data)
        self.vel_timestamps.append(time)
    
    def analyze_data(self):
        """수집된 GNSS 데이터 분석"""
        self.get_logger().info(f"데이터 수집 완료. 분석 시작...")
        
        # 타이머 중지 (한 번만 분석)
        self.timer.cancel()
        
        # 충분한 데이터가 있는지 확인
        if len(self.fix_data) < 5 or len(self.vel_data) < 5:
            self.get_logger().error("충분한 GNSS 데이터가 수집되지 않았습니다.")
            return
        
        # Fix 데이터 분석
        fix_result = self.analyze_fix_data()
        
        # Velocity 데이터 분석
        vel_result = self.analyze_vel_data()
        
        # 추천 파라미터 계산
        recommended_params = self.calculate_recommended_params(fix_result, vel_result)
        
        # 결과 저장
        result = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'fix_sample_count': len(self.fix_data),
            'vel_sample_count': len(self.vel_data),
            'fix_analysis': fix_result,
            'vel_analysis': vel_result,
            'recommended_params': recommended_params
        }
        
        # YAML 파일로 저장
        yaml_file = Path(self.output_dir) / 'gnss_accuracy_analysis.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        self.get_logger().info(f"분석 결과가 {yaml_file}에 저장되었습니다.")
        self.get_logger().info("추천 EKF 파라미터:")
        for key, value in recommended_params.items():
            self.get_logger().info(f"  {key}: {value}")
        
        # 노드 종료
        self.get_logger().info("분석 완료. 노드를 종료합니다.")
        rclpy.shutdown()
    
    def analyze_fix_data(self):
        """NavSatFix 데이터 분석"""
        # 데이터 추출
        latitudes = [item['latitude'] for item in self.fix_data]
        longitudes = [item['longitude'] for item in self.fix_data]
        altitudes = [item['altitude'] for item in self.fix_data]
        
        # 위치 통계 계산
        lat_mean = np.mean(latitudes)
        lon_mean = np.mean(longitudes)
        alt_mean = np.mean(altitudes)
        
        # 위치 편차 계산 (미터 단위로 변환)
        # 위도 1도는 약 111km, 경도 1도는 위도에 따라 다름 
        # (적도에서 약 111km, 위도가 높아질수록 감소)
        lat_dev_meters = np.array([(lat - lat_mean) * 111000 for lat in latitudes])
        lon_dev_meters = np.array([(lon - lon_mean) * 111000 * np.cos(np.radians(lat_mean)) 
                                  for lon in longitudes])
        alt_dev_meters = np.array([alt - alt_mean for alt in altitudes])
        
        # 위치 편차 통계
        ne_std = np.sqrt(np.mean(lat_dev_meters**2 + lon_dev_meters**2))  # 수평 방향 표준편차
        d_std = np.std(alt_dev_meters)  # 수직 방향 표준편차
        
        # 공분산 분석
        position_covariances = [item['position_covariance'] for item in self.fix_data 
                               if item['position_covariance_type'] > 0]
        
        if position_covariances:
            # 공분산 행렬에서 위도, 경도, 고도 분산 추출
            lat_vars = [cov[0] for cov in position_covariances]
            lon_vars = [cov[4] for cov in position_covariances]
            alt_vars = [cov[8] for cov in position_covariances]
            
            ne_cov_std = np.sqrt(np.mean([max(lat, lon) for lat, lon in zip(lat_vars, lon_vars)]))
            d_cov_std = np.sqrt(np.mean(alt_vars))
        else:
            # 공분산 정보가 없는 경우 위치 편차로 추정
            ne_cov_std = ne_std
            d_cov_std = d_std
        
        # 그래프 그리기
        self.plot_fix_data(lat_dev_meters, lon_dev_meters, alt_dev_meters)
        
        # 결과 반환
        return {
            'lat_mean': float(lat_mean),
            'lon_mean': float(lon_mean),
            'alt_mean': float(alt_mean),
            'horizontal_std_meters': float(ne_std),
            'vertical_std_meters': float(d_std),
            'horizontal_covariance_std': float(ne_cov_std),
            'vertical_covariance_std': float(d_cov_std)
        }
    
    def analyze_vel_data(self):
        """TwistWithCovarianceStamped 데이터 분석"""
        # 데이터 추출
        linear_x = [item['linear']['x'] for item in self.vel_data]
        linear_y = [item['linear']['y'] for item in self.vel_data]
        linear_z = [item['linear']['z'] for item in self.vel_data]
        
        # 속도 통계 계산
        x_mean = np.mean(linear_x)
        y_mean = np.mean(linear_y)
        z_mean = np.mean(linear_z)
        
        x_std = np.std(linear_x)
        y_std = np.std(linear_y)
        z_std = np.std(linear_z)
        
        # 수평 및 수직 속도 표준편차
        ne_std = np.sqrt(x_std**2 + y_std**2)  # 수평 방향
        d_std = z_std  # 수직 방향
        
        # 속도 크기 계산
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(linear_x, linear_y)]
        speed_mean = np.mean(speeds)
        speed_std = np.std(speeds)
        
        # 공분산 분석
        covariances = [np.array(item['covariance']).reshape(6, 6) for item in self.vel_data]
        
        if covariances:
            # 선속도에 대한 공분산 성분 추출 (x, y, z = 인덱스 0, 1, 2)
            x_vars = [cov[0, 0] for cov in covariances]
            y_vars = [cov[1, 1] for cov in covariances]
            z_vars = [cov[2, 2] for cov in covariances]
            
            ne_cov_std = np.sqrt(np.mean([max(x, y) for x, y in zip(x_vars, y_vars)]))
            d_cov_std = np.sqrt(np.mean(z_vars))
        else:
            # 공분산 정보가 없는 경우 속도 표준편차로 추정
            ne_cov_std = ne_std
            d_cov_std = d_std
        
        # 그래프 그리기
        self.plot_vel_data(linear_x, linear_y, linear_z, speeds)
        
        # 결과 반환
        return {
            'linear_x_mean': float(x_mean),
            'linear_y_mean': float(y_mean),
            'linear_z_mean': float(z_mean),
            'linear_x_std': float(x_std),
            'linear_y_std': float(y_std),
            'linear_z_std': float(z_std),
            'horizontal_vel_std': float(ne_std),
            'vertical_vel_std': float(d_std),
            'speed_mean': float(speed_mean),
            'speed_std': float(speed_std),
            'horizontal_vel_covariance_std': float(ne_cov_std),
            'vertical_vel_covariance_std': float(d_cov_std)
        }
    
    def calculate_recommended_params(self, fix_result, vel_result):
        """분석 결과를 바탕으로 추천 파라미터 계산"""
        # 정적 데이터에서는 GNSS 노이즈 요소만 계산
        # GNSS 위치 노이즈 - 실측 표준편차의 1.5배로 설정
        gps_pos_noise_ne = max(0.5, fix_result['horizontal_covariance_std'] * 1.5)
        gps_pos_noise_d = max(1.0, fix_result['vertical_covariance_std'] * 1.5)
        
        # GNSS 속도 노이즈 - 실측 표준편차의 1.5배로 설정
        gps_vel_noise_ne = max(0.1, vel_result['horizontal_vel_covariance_std'] * 1.5)
        gps_vel_noise_d = max(0.2, vel_result['vertical_vel_covariance_std'] * 1.5)
        
        # 추천 EKF 파라미터 반환
        return {
            'gps_pos_noise_ne': float(gps_pos_noise_ne),
            'gps_pos_noise_d': float(gps_pos_noise_d),
            'gps_vel_noise_ne': float(gps_vel_noise_ne),
            'gps_vel_noise_d': float(gps_vel_noise_d),
            'min_speed_for_gnss_heading': 0.5,  # 정적 데이터에서는 기본값 사용
            'use_gnss_heading': True  # 일반적으로 True 권장
        }
    
    def plot_fix_data(self, lat_dev, lon_dev, alt_dev):
        """GNSS 위치 데이터를 시각화하여 PNG 파일로 저장"""
        plt.figure(figsize=(12, 10))
        
        # 수평 위치 산점도
        plt.subplot(2, 1, 1)
        plt.scatter(lon_dev, lat_dev, c='blue', alpha=0.6)
        plt.title('GNSS 수평 위치 편차 (미터)')
        plt.xlabel('경도 편차 (m)')
        plt.ylabel('위도 편차 (m)')
        plt.grid(True)
        plt.axis('equal')
        
        # 위치 오차의 95% 신뢰 타원 그리기
        from matplotlib.patches import Ellipse
        std_dev_x = np.std(lon_dev)
        std_dev_y = np.std(lat_dev)
        ellipse = Ellipse((0, 0), width=std_dev_x*2*1.96, height=std_dev_y*2*1.96, 
                         edgecolor='red', facecolor='none', linestyle='--')
        plt.gca().add_patch(ellipse)
        plt.legend(['위치 샘플', '95% 신뢰 영역'])
        
        # 고도 시계열
        plt.subplot(2, 1, 2)
        rel_time = np.arange(len(alt_dev))
        plt.plot(rel_time, alt_dev, 'g-')
        plt.title('GNSS 고도 편차 (미터)')
        plt.xlabel('샘플 번호')
        plt.ylabel('고도 편차 (m)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        plt_file = Path(self.output_dir) / 'gnss_position_analysis.png'
        plt.savefig(str(plt_file))
        plt.close()
        
        self.get_logger().info(f"GNSS 위치 데이터 그래프가 {plt_file}에 저장되었습니다.")
    
    def plot_vel_data(self, vel_x, vel_y, vel_z, speeds):
        """GNSS 속도 데이터를 시각화하여 PNG 파일로 저장"""
        plt.figure(figsize=(12, 10))
        
        # 선속도 시계열
        plt.subplot(2, 1, 1)
        rel_time = np.arange(len(vel_x))
        plt.plot(rel_time, vel_x, 'r-', label='X')
        plt.plot(rel_time, vel_y, 'g-', label='Y')
        plt.plot(rel_time, vel_z, 'b-', label='Z')
        plt.title('GNSS 선속도 (m/s)')
        plt.xlabel('샘플 번호')
        plt.ylabel('속도 (m/s)')
        plt.grid(True)
        plt.legend()
        
        # 속도 크기 히스토그램
        plt.subplot(2, 1, 2)
        plt.hist(speeds, bins=30, color='green', alpha=0.7)
        plt.title('GNSS 속도 크기 분포')
        plt.xlabel('속도 (m/s)')
        plt.ylabel('빈도')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        plt_file = Path(self.output_dir) / 'gnss_velocity_analysis.png'
        plt.savefig(str(plt_file))
        plt.close()
        
        self.get_logger().info(f"GNSS 속도 데이터 그래프가 {plt_file}에 저장되었습니다.")


def main(args=None):
    parser = argparse.ArgumentParser(description='GNSS 정확도 분석 도구')
    parser.add_argument('-d', '--duration', type=float, default=60.0,
                        help='데이터 수집 기간(초), 기본값: 60초')
    parser.add_argument('-o', '--output-dir', type=str, default='ekf_param_results',
                        help='분석 결과 저장 디렉토리, 기본값: ekf_param_results')
    
    parsed_args = parser.parse_args(args=args)
    
    print(f"GNSS 정확도 분석 시작:")
    print(f"- 데이터 수집 기간: {parsed_args.duration}초")
    print(f"- 결과 저장 위치: {parsed_args.output_dir}")
    
    rclpy.init(args=args)
    node = GNSSAccuracyAnalyzer(
        duration=parsed_args.duration,
        output_dir=parsed_args.output_dir
    )
    rclpy.spin(node)
    
    # spin 종료 후 필요한 정리 작업
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
