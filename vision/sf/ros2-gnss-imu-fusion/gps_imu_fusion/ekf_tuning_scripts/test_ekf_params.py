#!/usr/bin/env python3
# test_ekf_params.py
# 생성된 EKF 파라미터 세트를 테스트하고 성능을 비교하는 스크립트

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import time
import threading
import signal
import json
import sys
import glob
import shutil

class EKFParameterTester(Node):
    def __init__(self, param_files, test_duration=60.0, output_dir='ekf_param_tests', test_name='test'):
        """여러 EKF 파라미터 세트를 테스트하는 노드
        
        Args:
            param_files: 테스트할 파라미터 파일 경로 리스트
            test_duration: 각 파라미터 세트별 테스트 기간(초)
            output_dir: 결과 저장 디렉토리
            test_name: 테스트 이름 (파일명에 사용)
        """
        super().__init__('ekf_parameter_tester')
        
        # 파라미터 설정
        self.param_files = param_files
        self.test_duration = test_duration
        self.output_dir = output_dir
        self.test_name = test_name
        
        # 결과 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장용 구조
        self.current_param_set = None
        self.test_results = {}
        
        # 구독자 설정
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose/filtered',
            self.pose_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',
            self.gps_callback,
            10
        )
        
        self.vel_sub = self.create_subscription(
            TwistWithCovarianceStamped,
            '/ublox_gps_node/fix_velocity',
            self.vel_callback,
            10
        )
        
        # 테스트 데이터
        self.current_data = {
            'poses': [],
            'odometry': [],
            'gps_fixes': [],
            'gps_velocities': [],
            'start_time': None
        }
        
        self.get_logger().info(f"EKF 파라미터 테스트 시작 - {len(param_files)}개 파라미터 세트 테스트")
    
    def pose_callback(self, msg):
        """필터링된 포즈 콜백"""
        if self.current_param_set and self.current_data['start_time']:
            # 시간 경과 계산
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - self.current_data['start_time']
            
            # 데이터 저장
            pose_data = {
                'timestamp': elapsed,
                'position': {
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.pose.orientation.x,
                    'y': msg.pose.pose.orientation.y,
                    'z': msg.pose.pose.orientation.z,
                    'w': msg.pose.pose.orientation.w
                },
                'covariance': list(msg.pose.covariance)
            }
            
            self.current_data['poses'].append(pose_data)
    
    def odom_callback(self, msg):
        """필터링된 오도메트리 콜백"""
        if self.current_param_set and self.current_data['start_time']:
            # 시간 경과 계산
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - self.current_data['start_time']
            
            # 데이터 저장
            odom_data = {
                'timestamp': elapsed,
                'position': {
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.pose.orientation.x,
                    'y': msg.pose.pose.orientation.y,
                    'z': msg.pose.pose.orientation.z,
                    'w': msg.pose.pose.orientation.w
                },
                'twist': {
                    'linear': {
                        'x': msg.twist.twist.linear.x,
                        'y': msg.twist.twist.linear.y,
                        'z': msg.twist.twist.linear.z
                    },
                    'angular': {
                        'x': msg.twist.twist.angular.x,
                        'y': msg.twist.twist.angular.y,
                        'z': msg.twist.twist.angular.z
                    }
                },
                'pose_covariance': list(msg.pose.covariance),
                'twist_covariance': list(msg.twist.covariance)
            }
            
            self.current_data['odometry'].append(odom_data)
    
    def gps_callback(self, msg):
        """GPS 위치 콜백"""
        if self.current_param_set and self.current_data['start_time']:
            # 시간 경과 계산
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - self.current_data['start_time']
            
            # 데이터 저장
            gps_data = {
                'timestamp': elapsed,
                'latitude': msg.latitude,
                'longitude': msg.longitude,
                'altitude': msg.altitude,
                'position_covariance': list(msg.position_covariance),
                'position_covariance_type': msg.position_covariance_type,
                'status': msg.status.status,
                'service': msg.status.service
            }
            
            self.current_data['gps_fixes'].append(gps_data)
    
    def vel_callback(self, msg):
        """GPS 속도 콜백"""
        if self.current_param_set and self.current_data['start_time']:
            # 시간 경과 계산
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - self.current_data['start_time']
            
            # 데이터 저장
            vel_data = {
                'timestamp': elapsed,
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
            
            self.current_data['gps_velocities'].append(vel_data)
    
    def test_parameter_set(self, param_file):
        """단일 파라미터 세트 테스트"""
        # 파라미터 파일 이름에서 세트 ID 추출
        param_set_id = os.path.splitext(os.path.basename(param_file))[0]
        self.get_logger().info(f"파라미터 세트 '{param_set_id}' 테스트 시작")
        
        # 현재 테스트 중인 파라미터 세트 설정
        self.current_param_set = param_set_id
        
        # 데이터 초기화
        self.current_data = {
            'poses': [],
            'odometry': [],
            'gps_fixes': [],
            'gps_velocities': [],
            'start_time': None
        }
        
        # EKF 노드 재실행 (파라미터 파일로 로드)
        try:
            # 여기서는 실제 노드 재실행 대신 로깅만 수행
            # 실제 구현에서는 subprocess를 사용하여 노드 재실행 필요
            self.get_logger().info(f"EKF 노드를 파라미터 파일 '{param_file}'로 재실행 (시뮬레이션)")
            
            # 실제 노드 재시작 코드 예시 (주석 처리)
            # cmd = ['ros2', 'run', 'your_package_name', 'ekf_fusion_node', '--ros-args', '--params-file', param_file]
            # proc = subprocess.Popen(cmd)
            
            # 데이터 수집 시작 시간 기록
            time.sleep(2)  # 노드가 시작되기를 기다림
            self.current_data['start_time'] = self.get_clock().now().nanoseconds / 1e9
            
            # 테스트 기간 동안 대기
            time.sleep(self.test_duration)
            
            # 노드 종료 (실제 구현 시 주석 해제)
            # proc.terminate()
            # proc.wait()
            
        except Exception as e:
            self.get_logger().error(f"EKF 노드 실행 중 오류 발생: {e}")
            return False
        
        # 수집된 데이터 분석
        analysis_result = self.analyze_test_data()
        
        # 결과 저장
        self.test_results[param_set_id] = {
            'param_file': param_file,
            'analysis': analysis_result,
            'data_count': {
                'poses': len(self.current_data['poses']),
                'odometry': len(self.current_data['odometry']),
                'gps_fixes': len(self.current_data['gps_fixes']),
                'gps_velocities': len(self.current_data['gps_velocities'])
            }
        }
        
        self.get_logger().info(f"파라미터 세트 '{param_set_id}' 테스트 완료")
        return True
    
    def analyze_test_data(self):
        """테스트 데이터 분석"""
        # 충분한 데이터가 수집되었는지 확인
        if (len(self.current_data['poses']) < 10 or 
            len(self.current_data['odometry']) < 10):
            self.get_logger().warning("충분한 테스트 데이터가 수집되지 않았습니다.")
            return {
                'success': False,
                'error': '충분한 데이터 없음'
            }
        
        # 분석 결과
        result = {
            'success': True,
            'trajectory_stats': self.analyze_trajectory(),
            'orientation_stats': self.analyze_orientation(),
            'velocity_stats': self.analyze_velocity(),
            'consistency_stats': self.analyze_consistency()
        }
        
        return result
    
    def analyze_trajectory(self):
        """경로 분석"""
        # 위치 데이터 추출
        positions_x = [data['position']['x'] for data in self.current_data['odometry']]
        positions_y = [data['position']['y'] for data in self.current_data['odometry']]
        
        # 통계 계산
        if positions_x and positions_y:
            total_distance = 0
            for i in range(1, len(positions_x)):
                dx = positions_x[i] - positions_x[i-1]
                dy = positions_y[i] - positions_y[i-1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            # 경로 시각화
            plt.figure(figsize=(10, 8))
            plt.plot(positions_x, positions_y, 'b-')
            plt.scatter(positions_x[0], positions_y[0], c='g', s=100, label='시작')
            plt.scatter(positions_x[-1], positions_y[-1], c='r', s=100, label='종료')
            plt.title(f'경로 궤적 - {self.current_param_set}')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_{self.current_param_set}_trajectory.png')
            plt.savefig(str(plot_file))
            plt.close()
            
            return {
                'total_distance': total_distance,
                'path_length': len(positions_x),
                'plot_file': str(plot_file)
            }
        else:
            return {
                'error': '궤적 데이터 없음'
            }
    
    def analyze_orientation(self):
        """자세 분석"""
        # 자세 데이터 추출 (쿼터니언 → 오일러)
        orientations = []
        for data in self.current_data['odometry']:
            qx = data['orientation']['x']
            qy = data['orientation']['y']
            qz = data['orientation']['z']
            qw = data['orientation']['w']
            
            # 쿼터니언 → 오일러 (롤, 피치, 요)
            roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)
            orientations.append((roll, pitch, yaw))
        
        if orientations:
            # 통계 계산
            orientations = np.array(orientations)
            roll_std = np.std(orientations[:, 0])
            pitch_std = np.std(orientations[:, 1])
            yaw_std = np.std(orientations[:, 2])
            
            # 시각화
            timestamps = [data['timestamp'] for data in self.current_data['odometry']]
            
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 1, 1)
            plt.plot(timestamps, np.degrees(orientations[:, 0]), 'r-')
            plt.title('롤 각도')
            plt.ylabel('각도 (도)')
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(timestamps, np.degrees(orientations[:, 1]), 'g-')
            plt.title('피치 각도')
            plt.ylabel('각도 (도)')
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(timestamps, np.degrees(orientations[:, 2]), 'b-')
            plt.title('요 각도 (방향)')
            plt.xlabel('시간 (초)')
            plt.ylabel('각도 (도)')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_{self.current_param_set}_orientation.png')
            plt.savefig(str(plot_file))
            plt.close()
            
            return {
                'roll_std_deg': np.degrees(roll_std),
                'pitch_std_deg': np.degrees(pitch_std),
                'yaw_std_deg': np.degrees(yaw_std),
                'plot_file': str(plot_file)
            }
        else:
            return {
                'error': '자세 데이터 없음'
            }
    
    def analyze_velocity(self):
        """속도 분석"""
        # 속도 데이터 추출
        linear_x = [data['twist']['linear']['x'] for data in self.current_data['odometry']]
        linear_y = [data['twist']['linear']['y'] for data in self.current_data['odometry']]
        linear_z = [data['twist']['linear']['z'] for data in self.current_data['odometry']]
        
        angular_z = [data['twist']['angular']['z'] for data in self.current_data['odometry']]
        
        if linear_x and linear_y:
            # 속도 크기 계산
            speeds = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(linear_x, linear_y, linear_z)]
            
            # 통계 계산
            mean_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            std_speed = np.std(speeds)
            
            mean_angular_z = np.mean(np.abs(angular_z))
            max_angular_z = np.max(np.abs(angular_z))
            
            # 시각화
            timestamps = [data['timestamp'] for data in self.current_data['odometry']]
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, speeds, 'b-')
            plt.title('선속도 크기')
            plt.ylabel('속도 (m/s)')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, angular_z, 'r-')
            plt.title('회전 속도 (Z축)')
            plt.xlabel('시간 (초)')
            plt.ylabel('각속도 (rad/s)')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_{self.current_param_set}_velocity.png')
            plt.savefig(str(plot_file))
            plt.close()
            
            return {
                'mean_speed': mean_speed,
                'max_speed': max_speed,
                'speed_std': std_speed,
                'mean_angular_z': mean_angular_z,
                'max_angular_z': max_angular_z,
                'plot_file': str(plot_file)
            }
        else:
            return {
                'error': '속도 데이터 없음'
            }
    
    def analyze_consistency(self):
        """측정값 일관성 분석"""
        # EKF 추정과 센서 측정값 비교
        # (실제 구현에서는 좌표계 변환 필요)
        
        # 간소화된 분석 - 포즈 공분산 추세 분석
        if self.current_data['poses']:
            # 위치 공분산 시계열 추출
            timestamps = [data['timestamp'] for data in self.current_data['poses']]
            
            # 위치 공분산 (X, Y, Z)
            pos_covs = []
            for data in self.current_data['poses']:
                cov_matrix = np.array(data['covariance']).reshape(6, 6)
                pos_covs.append([cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[2, 2]])
            
            pos_covs = np.array(pos_covs)
            
            # 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, pos_covs[:, 0], 'r-', label='X 위치')
            plt.plot(timestamps, pos_covs[:, 1], 'g-', label='Y 위치')
            plt.plot(timestamps, pos_covs[:, 2], 'b-', label='Z 위치')
            plt.title('위치 추정 공분산 추세')
            plt.xlabel('시간 (초)')
            plt.ylabel('공분산')
            plt.grid(True)
            plt.legend()
            plt.yscale('log')
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_{self.current_param_set}_consistency.png')
            plt.savefig(str(plot_file))
            plt.close()
            
            # 공분산 수렴 여부 평가
            # (낮은 값으로 수렴하는 것이 바람직)
            converged = all(pos_covs[-1, i] < pos_covs[0, i] for i in range(3))
            
            return {
                'converged': converged,
                'final_position_covariance': {
                    'x': float(pos_covs[-1, 0]),
                    'y': float(pos_covs[-1, 1]),
                    'z': float(pos_covs[-1, 2])
                },
                'plot_file': str(plot_file)
            }
        else:
            return {
                'error': '포즈 데이터 없음'
            }
    
    def quaternion_to_euler(self, x, y, z, w):
        """쿼터니언을 오일러 각도로 변환 (롤, 피치, 요)"""
        # 롤 (x축 회전)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # 피치 (y축 회전)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 90도 if out of range
        else:
            pitch = np.arcsin(sinp)
            
        # 요 (z축 회전)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def run_all_tests(self):
        """모든 파라미터 세트 테스트"""
        for param_file in self.param_files:
            if not os.path.exists(param_file):
                self.get_logger().error(f"파라미터 파일을 찾을 수 없습니다: {param_file}")
                continue
                
            self.test_parameter_set(param_file)
            
        # 테스트 결과 저장
        self.save_test_results()
        
        # 비교 분석
        self.compare_results()
    
    def save_test_results(self):
        """테스트 결과를 파일로 저장"""
        # 결과 파일 경로
        result_file = Path(self.output_dir) / f'{self.test_name}_test_results.json'
        
        # JSON으로 저장
        with open(str(result_file), 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        self.get_logger().info(f"테스트 결과가 {result_file}에 저장되었습니다.")
    
    def compare_results(self):
        """여러 파라미터 세트의 결과 비교"""
        # 비교 지표 추출
        comparison = {}
        
        for param_set, results in self.test_results.items():
            if 'analysis' not in results or not results['analysis']['success']:
                continue
                
            analysis = results['analysis']
            
            # 주요 지표 추출
            metrics = {}
            
            # 궤적
            if 'trajectory_stats' in analysis and 'error' not in analysis['trajectory_stats']:
                metrics['total_distance'] = analysis['trajectory_stats'].get('total_distance', 0)
            
            # 자세 안정성
            if 'orientation_stats' in analysis and 'error' not in analysis['orientation_stats']:
                metrics['roll_std_deg'] = analysis['orientation_stats'].get('roll_std_deg', 0)
                metrics['pitch_std_deg'] = analysis['orientation_stats'].get('pitch_std_deg', 0)
                metrics['yaw_std_deg'] = analysis['orientation_stats'].get('yaw_std_deg', 0)
            
            # 속도
            if 'velocity_stats' in analysis and 'error' not in analysis['velocity_stats']:
                metrics['mean_speed'] = analysis['velocity_stats'].get('mean_speed', 0)
                metrics['speed_std'] = analysis['velocity_stats'].get('speed_std', 0)
            
            # 일관성
            if 'consistency_stats' in analysis and 'error' not in analysis['consistency_stats']:
                metrics['converged'] = analysis['consistency_stats'].get('converged', False)
                if 'final_position_covariance' in analysis['consistency_stats']:
                    cov = analysis['consistency_stats']['final_position_covariance']
                    metrics['final_pos_cov_x'] = cov.get('x', 0)
                    metrics['final_pos_cov_y'] = cov.get('y', 0)
            
            comparison[param_set] = metrics
        
        # 비교 시각화
        if len(comparison) > 1:
            self.visualize_comparison(comparison)
    
    def visualize_comparison(self, comparison):
        """파라미터 세트 간 성능 비교 시각화"""
        param_sets = list(comparison.keys())
        
        # 주요 지표 그래프
        if all('roll_std_deg' in comparison[param] for param in param_sets):
            # 자세 안정성 비교
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(param_sets))
            width = 0.25
            
            plt.bar(x - width, [comparison[p]['roll_std_deg'] for p in param_sets], width, label='롤 표준편차')
            plt.bar(x, [comparison[p]['pitch_std_deg'] for p in param_sets], width, label='피치 표준편차')
            plt.bar(x + width, [comparison[p]['yaw_std_deg'] for p in param_sets], width, label='요 표준편차')
            
            plt.xlabel('파라미터 세트')
            plt.ylabel('표준편차 (도)')
            plt.title('자세 안정성 비교')
            plt.xticks(x, param_sets)
            plt.legend()
            plt.grid(True, axis='y')
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_orientation_comparison.png')
            plt.savefig(str(plot_file))
            plt.close()
        
        # 위치 공분산 비교
        if all('final_pos_cov_x' in comparison[param] for param in param_sets):
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(param_sets))
            width = 0.4
            
            plt.bar(x - width/2, [comparison[p]['final_pos_cov_x'] for p in param_sets], width, label='X 위치 공분산')
            plt.bar(x + width/2, [comparison[p]['final_pos_cov_y'] for p in param_sets], width, label='Y 위치 공분산')
            
            plt.xlabel('파라미터 세트')
            plt.ylabel('공분산')
            plt.title('위치 추정 정확도 비교 (낮을수록 좋음)')
            plt.xticks(x, param_sets)
            plt.legend()
            plt.grid(True, axis='y')
            plt.yscale('log')
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_position_accuracy_comparison.png')
            plt.savefig(str(plot_file))
            plt.close()
        
        # 속도 비교
        if all('mean_speed' in comparison[param] and 'speed_std' in comparison[param] for param in param_sets):
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(param_sets))
            width = 0.4
            
            plt.bar(x - width/2, [comparison[p]['mean_speed'] for p in param_sets], width, label='평균 속도')
            plt.bar(x + width/2, [comparison[p]['speed_std'] for p in param_sets], width, label='속도 표준편차')
            
            plt.xlabel('파라미터 세트')
            plt.ylabel('속도 (m/s)')
            plt.title('속도 추정 비교')
            plt.xticks(x, param_sets)
            plt.legend()
            plt.grid(True, axis='y')
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_velocity_comparison.png')
            plt.savefig(str(plot_file))
            plt.close()
            
        # 종합 점수 (낮을수록 좋음) - 예시: 자세 표준편차 + 위치 공분산 로그값
        scores = {}
        for param in param_sets:
            if all(key in comparison[param] for key in ['roll_std_deg', 'pitch_std_deg', 'yaw_std_deg', 'final_pos_cov_x', 'final_pos_cov_y']):
                # 간단한 가중 점수 계산 (낮을수록 좋음)
                attitude_score = comparison[param]['roll_std_deg'] + comparison[param]['pitch_std_deg'] + comparison[param]['yaw_std_deg']
                position_score = np.log10(comparison[param]['final_pos_cov_x'] + comparison[param]['final_pos_cov_y'])
                
                scores[param] = attitude_score * 0.3 + position_score * 0.7  # 가중치 적용
        
        if scores:
            # 스코어 기준 순위
            ranked_params = sorted(scores.keys(), key=lambda p: scores[p])
            
            # 시각화
            plt.figure(figsize=(12, 6))
            
            plt.bar(ranked_params, [scores[p] for p in ranked_params])
            
            plt.xlabel('파라미터 세트')
            plt.ylabel('종합 점수 (낮을수록 좋음)')
            plt.title('파라미터 세트 종합 성능 비교')
            plt.grid(True, axis='y')
            
            # 그래프 저장
            plot_file = Path(self.output_dir) / f'{self.test_name}_overall_score_comparison.png')
            plt.savefig(str(plot_file))
            plt.close()
            
            # 최적 파라미터 세트 선정
            best_param = ranked_params[0] if ranked_params else None
            
            if best_param:
                self.get_logger().info(f"테스트 결과 기준 최적 파라미터 세트: {best_param}")
                
                # 최적 파라미터 세트 복사
                best_file = self.test_results[best_param]['param_file']
                dest_file = Path(self.output_dir) / f'{self.test_name}_best_params.yaml'
                
                # 파일 복사
                shutil.copy2(best_file, str(dest_file))
                
                self.get_logger().info(f"최적 파라미터 세트가 {dest_file}에 저장되었습니다.")
                
                # 요약 파일 생성
                summary = {
                    'test_name': self.test_name,
                    'test_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'tested_param_sets': param_sets,
                    'best_param_set': best_param,
                    'scores': scores,
                    'best_param_file': str(dest_file)
                }
                
                summary_file = Path(self.output_dir) / f'{self.test_name}_test_summary.yaml'
                with open(str(summary_file), 'w') as f:
                    yaml.dump(summary, f, default_flow_style=False)
                
                self.get_logger().info(f"테스트 요약이 {summary_file}에 저장되었습니다.")


def find_parameter_files(directory, pattern='ekf_fusion_params*.yaml'):
    """디렉토리에서 패턴과 일치하는 파라미터 파일 찾기"""
    return list(Path(directory).glob(pattern))


def main(args=None):
    parser = argparse.ArgumentParser(description='EKF 파라미터 테스트 도구')
    parser.add_argument('-p', '--param-files', type=str, nargs='+',
                        help='테스트할 파라미터 파일 경로 목록')
    parser.add_argument('-d', '--param-dir', type=str,
                        help='파라미터 파일을 찾을 디렉토리')
    parser.add_argument('-t', '--test-duration', type=float, default=60.0,
                        help='각 파라미터 세트별 테스트 기간(초), 기본값: 60초')
    parser.add_argument('-o', '--output-dir', type=str, default='ekf_param_tests',
                        help='결과 저장 디렉토리, 기본값: ekf_param_tests')
    parser.add_argument('-n', '--test-name', type=str, default='test',
                        help='테스트 이름 (파일명에 사용)')
    
    parsed_args = parser.parse_args(args=args)
    
    # 파라미터 파일 목록 결정
    param_files = []
    
    if parsed_args.param_files:
        param_files = parsed_args.param_files
    elif parsed_args.param_dir:
        param_files = find_parameter_files(parsed_args.param_dir)
    else:
        print("오류: 파라미터 파일(-p) 또는 파라미터 디렉토리(-d)를 지정해야 합니다.")
        return 1
    
    if not param_files:
        print("오류: 테스트할 파라미터 파일이 없습니다.")
        return 1
    
    print(f"EKF 파라미터 테스트 시작:")
    print(f"- 테스트할 파라미터 파일: {len(param_files)}개")
    for f in param_files:
        print(f"  - {f}")
    print(f"- 각 테스트 기간: {parsed_args.test_duration}초")
    print(f"- 결과 저장 위치: {parsed_args.output_dir}")
    
    rclpy.init(args=args)
    
    # 테스트 노드 실행
    tester = EKFParameterTester(
        param_files=param_files,
        test_duration=parsed_args.test_duration,
        output_dir=parsed_args.output_dir,
        test_name=parsed_args.test_name
    )
    
    # 시그널 핸들러 설정 (Ctrl+C 처리)
    def signal_handler(sig, frame):
        print("테스트 중단. 정리 중...")
        tester.save_test_results()
        tester.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 별도 스레드에서 테스트 실행
    test_thread = threading.Thread(target=tester.run_all_tests)
    test_thread.daemon = True
    test_thread.start()
    
    # 메인 스레드에서 ROS 스핀
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        # 정리
        tester.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
