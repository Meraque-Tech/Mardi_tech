#!/usr/bin/env python3
# generate_ekf_params.py
# 분석 결과들을 종합하여 최적의 EKF 파라미터 YAML 파일을 생성하는 스크립트

import rclpy
from rclpy.node import Node
import yaml
import os
from pathlib import Path
from datetime import datetime
import argparse
import glob

class EKFParamGenerator:
    def __init__(self, input_dir='ekf_param_results', output_dir='ekf_param_results'):
        """분석 결과로부터 EKF 파라미터를 생성하는 클래스
        
        Args:
            input_dir: 분석 결과 디렉토리
            output_dir: 생성된 파라미터를 저장할 디렉토리
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 결과 디렉토리 확인
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 기본 파라미터 (사용자 정의 파라미터로 덮어쓰이지 않는 값들)
        self.default_params = {
            'world_frame_id': 'map',
            'base_frame_id': 'base_link',
            'gnss_frame_id': 'gps',
            'imu_frame_id': 'imu_link',
            'update_rate': 50.0,
            'publish_tf': True,
            'mag_declination': -8.0,
            'use_magnetic_declination': False,
            'init_pos_unc': 10.0,
            'init_vel_unc': 1.0,
            'init_att_unc': 0.34906,
            'init_hdg_unc': 3.14159,
            'init_accel_bias_unc': 0.981,
            'init_gyro_bias_unc': 0.01745
        }
    
    def load_analysis_results(self):
        """분석 결과를 로드하고 종합"""
        imu_noise_file = os.path.join(self.input_dir, 'imu_noise_analysis.yaml')
        gnss_accuracy_file = os.path.join(self.input_dir, 'gnss_accuracy_analysis.yaml')
        imu_bias_file = os.path.join(self.input_dir, 'imu_bias_analysis.yaml')
        
        imu_noise_results = self.load_yaml_file(imu_noise_file)
        gnss_accuracy_results = self.load_yaml_file(gnss_accuracy_file)
        imu_bias_results = self.load_yaml_file(imu_bias_file)
        
        return imu_noise_results, gnss_accuracy_results, imu_bias_results
    
    def load_yaml_file(self, file_path):
        """YAML 파일을 로드"""
        if not os.path.exists(file_path):
            print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
            return None
        
        with open(file_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                return data
            except yaml.YAMLError as e:
                print(f"YAML 파일 로드 중 오류 발생: {e}")
                return None
    
    def generate_params(self):
        """EKF 파라미터 생성"""
        # 분석 결과 로드
        imu_noise_results, gnss_accuracy_results, imu_bias_results = self.load_analysis_results()
        
        # 기본 파라미터로 시작
        ekf_params = self.default_params.copy()
        
        # IMU 노이즈 결과 적용
        if imu_noise_results and 'recommended_params' in imu_noise_results:
            params = imu_noise_results['recommended_params']
            ekf_params.update({
                'accel_noise': params.get('accel_noise', 0.05),
                'gyro_noise': params.get('gyro_noise', 0.00175)
            })
            print(f"IMU 노이즈 파라미터 적용: accel_noise={ekf_params['accel_noise']:.6f}, gyro_noise={ekf_params['gyro_noise']:.6f}")
        else:
            print("IMU 노이즈 분석 결과를 찾을 수 없어 기본값 사용")
            ekf_params.update({
                'accel_noise': 0.05,
                'gyro_noise': 0.00175
            })
        
        # IMU 바이어스 결과 적용
        if imu_bias_results and 'recommended_params' in imu_bias_results:
            params = imu_bias_results['recommended_params']
            ekf_params.update({
                'accel_bias_noise': params.get('accel_bias_noise', 0.01),
                'gyro_bias_noise': params.get('gyro_bias_noise', 0.00025),
                'accel_bias_tau': params.get('accel_bias_tau', 100.0),
                'gyro_bias_tau': params.get('gyro_bias_tau', 50.0)
            })
            print(f"IMU 바이어스 파라미터 적용: accel_bias_noise={ekf_params['accel_bias_noise']:.6f}, " + 
                 f"gyro_bias_noise={ekf_params['gyro_bias_noise']:.6f}, " +
                 f"accel_bias_tau={ekf_params['accel_bias_tau']:.1f}, " +
                 f"gyro_bias_tau={ekf_params['gyro_bias_tau']:.1f}")
        else:
            print("IMU 바이어스 분석 결과를 찾을 수 없어 기본값 사용")
            ekf_params.update({
                'accel_bias_noise': 0.01,
                'gyro_bias_noise': 0.00025,
                'accel_bias_tau': 100.0,
                'gyro_bias_tau': 50.0
            })
        
        # GNSS 정확도 결과 적용
        if gnss_accuracy_results and 'recommended_params' in gnss_accuracy_results:
            params = gnss_accuracy_results['recommended_params']
            ekf_params.update({
                'gps_pos_noise_ne': params.get('gps_pos_noise_ne', 0.2),
                'gps_pos_noise_d': params.get('gps_pos_noise_d', 1.0),
                'gps_vel_noise_ne': params.get('gps_vel_noise_ne', 0.5),
                'gps_vel_noise_d': params.get('gps_vel_noise_d', 1.0),
                'min_speed_for_gnss_heading': params.get('min_speed_for_gnss_heading', 0.5),
                'use_gnss_heading': params.get('use_gnss_heading', True)
            })
            print(f"GNSS 정확도 파라미터 적용: " +
                 f"gps_pos_noise_ne={ekf_params['gps_pos_noise_ne']:.6f}, " +
                 f"gps_pos_noise_d={ekf_params['gps_pos_noise_d']:.6f}, " +
                 f"gps_vel_noise_ne={ekf_params['gps_vel_noise_ne']:.6f}, " +
                 f"gps_vel_noise_d={ekf_params['gps_vel_noise_d']:.6f}")
        else:
            print("GNSS 정확도 분석 결과를 찾을 수 없어 기본값 사용")
            ekf_params.update({
                'gps_pos_noise_ne': 0.2,
                'gps_pos_noise_d': 1.0,
                'gps_vel_noise_ne': 0.5,
                'gps_vel_noise_d': 1.0,
                'min_speed_for_gnss_heading': 0.5,
                'use_gnss_heading': True
            })
        
        return ekf_params
    
    def generate_yaml(self, ekf_params, filename='ekf_fusion_params.yaml'):
        """EKF 파라미터를 YAML 파일로 생성"""
        # YAML 파일 구조 생성
        ros_params = {
            'ekf_fusion_node': {
                'ros__parameters': ekf_params
            }
        }
        
        # 파일 경로 생성
        file_path = os.path.join(self.output_dir, filename)
        
        # YAML 파일로 저장
        with open(file_path, 'w') as f:
            # 주석 추가
            f.write("# EKF 퓨전 노드 파라미터\n")
            f.write(f"# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# 이 파일은 자동으로 생성되었습니다.\n\n")
            
            yaml.dump(ros_params, f, default_flow_style=False)
        
        print(f"EKF 파라미터가 {file_path}에 저장되었습니다.")
        return file_path
    
    def process(self, output_filename='ekf_fusion_params.yaml'):
        """전체 처리 과정"""
        # EKF 파라미터 생성
        ekf_params = self.generate_params()
        
        # YAML 파일 생성
        yaml_path = self.generate_yaml(ekf_params, output_filename)
        
        return yaml_path


def find_latest_results_dir():
    """가장 최근의 결과 디렉토리 찾기"""
    dirs = glob.glob('ekf_param_results*')
    if not dirs:
        return 'ekf_param_results'
    
    # 디렉토리가 존재하면 날짜/시간 접미사를 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'ekf_param_results_{timestamp}'


def main(args=None):
    parser = argparse.ArgumentParser(description='EKF 파라미터 생성 도구')
    parser.add_argument('-i', '--input-dir', type=str, default='ekf_param_results',
                        help='분석 결과 디렉토리, 기본값: ekf_param_results')
    parser.add_argument('-o', '--output-dir', type=str,
                        help='출력 디렉토리, 기본값: 입력 디렉토리와 동일')
    parser.add_argument('-f', '--filename', type=str, default='ekf_fusion_params.yaml',
                        help='출력 파일 이름, 기본값: ekf_fusion_params.yaml')
    
    parsed_args = parser.parse_args(args=args)
    
    # 입력 디렉토리 확인
    input_dir = parsed_args.input_dir
    if not os.path.exists(input_dir):
        print(f"경고: 입력 디렉토리 {input_dir}가 존재하지 않습니다. 디렉토리를 생성합니다.")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
    
    # 출력 디렉토리가 지정되지 않았으면 입력 디렉토리와 동일하게 설정
    output_dir = parsed_args.output_dir if parsed_args.output_dir else input_dir
    
    print(f"EKF 파라미터 생성 시작:")
    print(f"- 입력 디렉토리: {input_dir}")
    print(f"- 출력 디렉토리: {output_dir}")
    print(f"- 출력 파일 이름: {parsed_args.filename}")
    
    generator = EKFParamGenerator(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    yaml_path = generator.process(output_filename=parsed_args.filename)
    print(f"EKF 파라미터 생성 완료: {yaml_path}")


if __name__ == '__main__':
    main()
