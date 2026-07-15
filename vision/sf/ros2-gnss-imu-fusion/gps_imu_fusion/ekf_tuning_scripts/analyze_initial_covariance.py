#!/usr/bin/env python3
# analyze_initial_covariance.py
# EKF 필터 초기 공분산 파라미터 분석 및 최적화

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import NavSatFix, Imu
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import yaml
import time
import argparse
import subprocess
import signal
import sys
from datetime import datetime


class InitialCovarianceAnalyzer(Node):
    def __init__(self, duration=60.0, static_mode=True, output_dir='ekf_param_results', 
                 restart_cmd=None, num_trials=3):
        """EKF 초기 공분산 분석 노드
        
        Args:
            duration: 각 시도당 분석 시간(초)
            static_mode: True=정지 상태 분석, False=동적 상태 분석
            output_dir: 결과 저장 디렉토리
            restart_cmd: EKF 노드 재시작 명령 (None=재시작 안함)
            num_trials: 시도 횟수
        """
        super().__init__('initial_covariance_analyzer')
        
        # 파라미터 설정
        self.duration = duration
        self.static_mode = static_mode
        self.output_dir = output_dir
        self.restart_cmd = restart_cmd
        self.num_trials = num_trials
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 분석 모드 문자열 (로그용)
        self.mode_str = "정적(정지)" if static_mode else "동적(이동)"
        
        # 데이터 저장
        self.data = {
            'gps': [],
            'imu': [],
            'poses': [],
            'odom': []
        }
        
        # 수렴 시간 저장
        self.convergence_times = {
            'position': [],
            'velocity': [],
            'attitude': [],
            'heading': []
        }
        
        # 토픽 구독
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',
            self.gps_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
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
        
        # 타이머 초기화
        self.start_time = None
        self.trial_count = 0
        
        self.get_logger().info(f"초기 공분산 분석 시작 - {self.mode_str} 모드, {self.num_trials}회 시도")
        
        # 첫 번째 시도 시작
        self.start_next_trial()
    
    def start_next_trial(self):
        """다음 시도 시작"""
        if self.trial_count >= self.num_trials:
            # 모든 시도 완료
            self.analyze_results()
            return
        
        self.trial_count += 1
        self.get_logger().info(f"시도 {self.trial_count}/{self.num_trials} 시작")
        
        # 데이터 초기화
        self.data = {
            'gps': [],
            'imu': [],
            'poses': [],
            'odom': []
        }
        
        # EKF 재시작 (명령이 제공된 경우)
        if self.restart_cmd:
            try:
                self.get_logger().info(f"EKF 노드 재시작: {self.restart_cmd}")
                subprocess.run(self.restart_cmd, shell=True)
                time.sleep(2)  # 노드 시작 대기
            except Exception as e:
                self.get_logger().error(f"EKF 노드 재시작 실패: {e}")
        
        # 시작 시간 설정
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        # 타이머 설정 (지정된 시간 후 시도 종료)
        self.timer = self.create_timer(self.duration, self.end_current_trial)
    
    def end_current_trial(self):
        """현재 시도 종료"""
        self.timer.cancel()
        
        # 수렴 시간 계산
        self.calculate_convergence()
        
        # 다음 시도 시작
        self.start_next_trial()
    
    def calculate_convergence(self):
        """현재 시도의 수렴 시간 계산"""
        if not self.data['poses'] or not self.data['odom']:
            self.get_logger().warning("충분한 데이터가 없어 수렴 시간을 계산할 수 없습니다.")
            return
        
        # 위치 공분산 수렴 시간 계산
        pos_cov_time = self.calculate_position_convergence()
        if pos_cov_time:
            self.convergence_times['position'].append(pos_cov_time)
            self.get_logger().info(f"위치 공분산 수렴 시간: {pos_cov_time:.2f}초")
        
        # 속도 공분산 수렴 시간 계산
        vel_cov_time = self.calculate_velocity_convergence()
        if vel_cov_time:
            self.convergence_times['velocity'].append(vel_cov_time)
            self.get_logger().info(f"속도 공분산 수렴 시간: {vel_cov_time:.2f}초")
        
        # 자세 공분산 수렴 시간 계산
        att_cov_time = self.calculate_attitude_convergence()
        if att_cov_time:
            self.convergence_times['attitude'].append(att_cov_time)
            self.get_logger().info(f"자세 공분산 수렴 시간: {att_cov_time:.2f}초")
        
        # 방향(헤딩) 공분산 수렴 시간 계산
        hdg_cov_time = self.calculate_heading_convergence()
        if hdg_cov_time:
            self.convergence_times['heading'].append(hdg_cov_time)
            self.get_logger().info(f"방향 공분산 수렴 시간: {hdg_cov_time:.2f}초")
    
    def calculate_position_convergence(self):
        """위치 공분산 수렴 시간 계산"""
        # 위치 공분산 시계열 추출 (X, Y, Z 평균)
        pos_covs = []
        timestamps = []
        
        for i, data in enumerate(self.data['poses']):
            time = data['timestamp']
            cov = data['covariance']
            
            # 위치 공분산 (인덱스 0, 7, 14는 X, Y, Z 위치 분산)
            pos_cov_avg = (cov[0] + cov[7] + cov[14]) / 3
            
            pos_covs.append(pos_cov_avg)
            timestamps.append(time)
        
        return self.find_convergence_time(timestamps, pos_covs)
    
    def calculate_velocity_convergence(self):
        """속도 공분산 수렴 시간 계산"""
        # 속도 공분산 시계열 추출
        vel_covs = []
        timestamps = []
        
        for i, data in enumerate(self.data['odom']):
            time = data['timestamp']
            twist_cov = data['twist_covariance']
            
            # 선속도 공분산 (인덱스 0, 7, 14는 X, Y, Z 속도 분산)
            vel_cov_avg = (twist_cov[0] + twist_cov[7] + twist_cov[14]) / 3
            
            vel_covs.append(vel_cov_avg)
            timestamps.append(time)
        
        return self.find_convergence_time(timestamps, vel_covs)
    
    def calculate_attitude_convergence(self):
        """자세(롤/피치) 공분산 수렴 시간 계산"""
        # 자세 공분산 시계열 추출
        att_covs = []
        timestamps = []
        
        for i, data in enumerate(self.data['poses']):
            time = data['timestamp']
            cov = data['covariance']
            
            # 롤/피치 공분산 (인덱스 21, 28는 롤/피치 분산)
            att_cov_avg = (cov[21] + cov[28]) / 2
            
            att_covs.append(att_cov_avg)
            timestamps.append(time)
        
        return self.find_convergence_time(timestamps, att_covs)
    
    def calculate_heading_convergence(self):
        """방향(헤딩) 공분산 수렴 시간 계산"""
        # 헤딩 공분산 시계열 추출
        hdg_covs = []
        timestamps = []
        
        for i, data in enumerate(self.data['poses']):
            time = data['timestamp']
            cov = data['covariance']
            
            # 헤딩 공분산 (인덱스 35는 요(yaw) 분산)
            hdg_cov = cov[35]
            
            hdg_covs.append(hdg_cov)
            timestamps.append(time)
        
        return self.find_convergence_time(timestamps, hdg_covs)
    
    def find_convergence_time(self, timestamps, cov_values, threshold_pct=0.1):
        """공분산 시계열에서 수렴 시간 찾기
        
        Args:
            timestamps: 시간 배열
            cov_values: 공분산 값 배열
            threshold_pct: 수렴 기준 (최종값 대비 비율)
        
        Returns:
            float: 수렴 시간 (초)
        """
        if not timestamps or not cov_values or len(timestamps) < 10:
            return None
        
        # 최종 공분산의 (1 + threshold_pct) 배 지점을 수렴 임계값으로 설정
        final_cov = cov_values[-1]
        threshold = final_cov * (1 + threshold_pct)
        
        # 임계값 이하로 감소한 첫 시점 찾기
        for i, cov in enumerate(cov_values):
            if i > 5 and cov <= threshold:  # 처음 몇 개는 건너뛰기
                return timestamps[i]
        
        # 수렴 지점을 찾지 못한 경우
        return None
    
    def gps_callback(self, msg):
        """GPS 메시지 콜백"""
        if self.start_time is None:
            return
            
        time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        
        # GPS 데이터 저장
        data = {
            'timestamp': time,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'covariance': list(msg.position_covariance),
            'covariance_type': msg.position_covariance_type
        }
        
        self.data['gps'].append(data)
    
    def imu_callback(self, msg):
        """IMU 메시지 콜백"""
        if self.start_time is None:
            return
            
        time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        
        # IMU 데이터 저장
        data = {
            'timestamp': time,
            'orientation': {
                'x': msg.orientation.x,
                'y': msg.orientation.y,
                'z': msg.orientation.z,
                'w': msg.orientation.w
            },
            'angular_velocity': {
                'x': msg.angular_velocity.x,
                'y': msg.angular_velocity.y,
                'z': msg.angular_velocity.z
            },
            'linear_acceleration': {
                'x': msg.linear_acceleration.x,
                'y': msg.linear_acceleration.y,
                'z': msg.linear_acceleration.z
            }
        }
        
        self.data['imu'].append(data)
    
    def pose_callback(self, msg):
        """Pose 메시지 콜백"""
        if self.start_time is None:
            return
            
        time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        
        # Pose 데이터 저장
        data = {
            'timestamp': time,
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
        
        self.data['poses'].append(data)
    
    def odom_callback(self, msg):
        """Odometry 메시지 콜백"""
        if self.start_time is None:
            return
            
        time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        
        # Odometry 데이터 저장
        data = {
            'timestamp': time,
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
        
        self.data['odom'].append(data)
    
    def analyze_results(self):
        """모든 시도의 결과 분석"""
        self.get_logger().info("모든 시도 완료. 결과 분석 중...")
        
        # 시도별 결과 확인
        if (not self.convergence_times['position'] or 
            not self.convergence_times['velocity'] or 
            not self.convergence_times['attitude'] or 
            not self.convergence_times['heading']):
            self.get_logger().warning("충분한 시도 결과가 없어 분석을 진행할 수 없습니다.")
        
        # 통계 계산
        stats = {}
        for key, times in self.convergence_times.items():
            if times:
                stats[key] = {
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'times': times
                }
        
        # 권장 초기 공분산 계산
        recommended_params = self.calculate_recommended_params(stats)
        
        # 시각화
        self.plot_convergence_times(stats)
        
        # 결과 저장
        result = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mode': 'static' if self.static_mode else 'dynamic',
            'num_trials': self.trial_count,
            'successful_trials': {
                'position': len(self.convergence_times['position']),
                'velocity': len(self.convergence_times['velocity']),
                'attitude': len(self.convergence_times['attitude']),
                'heading': len(self.convergence_times['heading'])
            },
            'convergence_stats': stats,
            'recommended_params': recommended_params
        }
        
        # YAML 파일로 저장
        mode_str = 'static' if self.static_mode else 'dynamic'
        result_file = Path(self.output_dir) / f'initial_covariance_{mode_str}_analysis.yaml'
        with open(result_file, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        self.get_logger().info(f"분석 결과가 {result_file}에 저장되었습니다.")
        self.get_logger().info("추천 초기 공분산 파라미터:")
        for key, value in recommended_params.items():
            self.get_logger().info(f"  {key}: {value}")
        
        # 노드 종료
        self.get_logger().info("분석 완료. 노드를 종료합니다.")
        rclpy.shutdown()
    
    def calculate_recommended_params(self, stats):
        """통계를 바탕으로 최적의 초기 공분산 파라미터 계산"""
        # 전략: 수렴 시간이 짧을수록 초기 불확실성은 실제에 가깝게(작게) 설정
        # 수렴 시간이 길수록 더 큰 불확실성 필요
        
        # 기본 파라미터 (합리적인 기본값)
        default_params = {
            'init_pos_unc': 10.0 if self.static_mode else 30.0,
            'init_vel_unc': 0.5 if self.static_mode else 3.0,
            'init_att_unc': 0.34906,  # ~20도
            'init_hdg_unc': 3.14159,  # ~180도
            'init_accel_bias_unc': 0.981,
            'init_gyro_bias_unc': 0.01745
        }
        
        # 수렴 시간이 없는 경우 기본값 사용
        if not stats:
            return default_params
        
        recommended = default_params.copy()
        
        # 위치 불확실성
        if 'position' in stats:
            pos_time = stats['position']['mean']
            # 빠른 수렴(1초 이하)일 경우 불확실성 감소, 느린 수렴일 경우 증가
            if pos_time < 1.0:
                recommended['init_pos_unc'] = 5.0 if self.static_mode else 15.0
            elif pos_time > 3.0:
                recommended['init_pos_unc'] = 20.0 if self.static_mode else 50.0
        
        # 속도 불확실성
        if 'velocity' in stats:
            vel_time = stats['velocity']['mean']
            if vel_time < 1.0:
                recommended['init_vel_unc'] = 0.2 if self.static_mode else 1.0
            elif vel_time > 3.0:
                recommended['init_vel_unc'] = 1.0 if self.static_mode else 5.0
        
        # 자세 불확실성
        if 'attitude' in stats:
            att_time = stats['attitude']['mean']
            if att_time < 1.0:
                recommended['init_att_unc'] = 0.17453  # ~10도
            elif att_time > 3.0:
                recommended['init_att_unc'] = 0.52360  # ~30도
        
        # 헤딩 불확실성 (방향)
        if 'heading' in stats:
            hdg_time = stats['heading']['mean']
            # 헤딩은 특별함: 너무 작게 설정하면 잘못된 방향으로 수렴할 수 있음
            if hdg_time < 2.0:
                # 빨리 수렴해도 방향이 확실하다면 불확실성 감소 가능
                recommended['init_hdg_unc'] = 1.57080  # ~90도
        
        return recommended
    
    def plot_convergence_times(self, stats):
        """수렴 시간 시각화"""
        if not stats:
            return
        
        # 막대 그래프: 상태별 평균 수렴 시간
        plt.figure(figsize=(10, 6))
        
        categories = list(stats.keys())
        means = [stats[cat]['mean'] for cat in categories]
        std_devs = [stats[cat]['std'] for cat in categories]
        
        bars = plt.bar(categories, means, yerr=std_devs, capsize=5)
        
        # 그래프 설정
        plt.title(f'EKF 상태별 평균 수렴 시간 ({self.mode_str} 모드)')
        plt.ylabel('수렴 시간 (초)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 데이터 레이블 추가
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{mean:.2f}s', ha='center', va='bottom')
        
        # 저장
        mode_str = 'static' if self.static_mode else 'dynamic'
        plot_file = Path(self.output_dir) / f'initial_covariance_{mode_str}_convergence.png'
        plt.tight_layout()
        plt.savefig(str(plot_file))
        plt.close()
        
        self.get_logger().info(f"수렴 시간 그래프가 {plot_file}에 저장되었습니다.")


def main(args=None):
    parser = argparse.ArgumentParser(description='초기 공분산 분석 도구')
    parser.add_argument('-d', '--duration', type=float, default=60.0,
                        help='각 시도당 분석 시간(초), 기본값: 60초')
    parser.add_argument('-s', '--static', action='store_true',
                        help='정적(정지) 모드로 분석 (기본: 동적)')
    parser.add_argument('-n', '--num-trials', type=int, default=3,
                        help='시도 횟수, 기본값: 3')
    parser.add_argument('-o', '--output-dir', type=str, default='ekf_param_results',
                        help='분석 결과 저장 디렉토리, 기본값: ekf_param_results')
    parser.add_argument('-r', '--restart-cmd', type=str,
                        help='EKF 노드 재시작 명령 (지정하지 않으면 재시작 안함)')
    
    parsed_args = parser.parse_args(args=args)
    
    print(f"초기 공분산 분석 시작:")
    print(f"- 모드: {'정적(정지)' if parsed_args.static else '동적(이동)'}")
    print(f"- 각 시도당 시간: {parsed_args.duration}초")
    print(f"- 시도 횟수: {parsed_args.num_trials}")
    print(f"- 결과 저장 위치: {parsed_args.output_dir}")
    
    if parsed_args.restart_cmd:
        print(f"- EKF 재시작 명령: {parsed_args.restart_cmd}")
    
    rclpy.init(args=args)
    
    node = InitialCovarianceAnalyzer(
        duration=parsed_args.duration,
        static_mode=parsed_args.static,
        output_dir=parsed_args.output_dir,
        restart_cmd=parsed_args.restart_cmd,
        num_trials=parsed_args.num_trials
    )
    
    # 시그널 핸들러 설정 (Ctrl+C 처리)
    def signal_handler(sig, frame):
        print("\n분석 중단. 정리 중...")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
