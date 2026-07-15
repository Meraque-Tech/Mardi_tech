#!/usr/bin/env python3
# analyze_test_results.py
# 테스트 결과 분석 및 최적 파라미터 선택 스크립트

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
from tabulate import tabulate
import glob
import sys


class TestResultAnalyzer:
    def __init__(self, result_file, output_dir=None):
        """테스트 결과 분석 클래스
        
        Args:
            result_file: 테스트 결과 JSON 파일 경로
            output_dir: 분석 결과 저장 디렉토리 (기본값: result_file과 동일 디렉토리)
        """
        self.result_file = result_file
        
        # 출력 디렉토리 설정
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.dirname(result_file)
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 결과 데이터 로드
        self.results = self.load_results()
        
        # 보고서 데이터
        self.report_data = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'result_file': result_file,
            'metrics': {},
            'rankings': {},
            'plots': {},
            'best_params': {}
        }
    
    def load_results(self):
        """결과 JSON 파일 로드"""
        try:
            with open(self.result_file, 'r') as f:
                results = json.load(f)
            return results
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"결과 파일 로드 중 오류 발생: {e}")
            return {}
    
    def extract_metrics(self):
        """각 파라미터 세트에서 메트릭 추출"""
        metrics = {}
        
        for param_set, data in self.results.items():
            if 'analysis' not in data or not data['analysis']['success']:
                continue
            
            analysis = data['analysis']
            param_metrics = {}
            
            # 궤적 메트릭
            if 'trajectory_stats' in analysis and 'error' not in analysis['trajectory_stats']:
                traj = analysis['trajectory_stats']
                param_metrics.update({
                    'total_distance': traj.get('total_distance', 0),
                    'path_length': traj.get('path_length', 0)
                })
            
            # 자세 메트릭
            if 'orientation_stats' in analysis and 'error' not in analysis['orientation_stats']:
                orient = analysis['orientation_stats']
                param_metrics.update({
                    'roll_std_deg': orient.get('roll_std_deg', 0),
                    'pitch_std_deg': orient.get('pitch_std_deg', 0),
                    'yaw_std_deg': orient.get('yaw_std_deg', 0)
                })
            
            # 속도 메트릭
            if 'velocity_stats' in analysis and 'error' not in analysis['velocity_stats']:
                vel = analysis['velocity_stats']
                param_metrics.update({
                    'mean_speed': vel.get('mean_speed', 0),
                    'max_speed': vel.get('max_speed', 0),
                    'speed_std': vel.get('speed_std', 0),
                    'mean_angular_z': vel.get('mean_angular_z', 0),
                    'max_angular_z': vel.get('max_angular_z', 0)
                })
            
            # 일관성 메트릭
            if 'consistency_stats' in analysis and 'error' not in analysis['consistency_stats']:
                cons = analysis['consistency_stats']
                param_metrics.update({
                    'converged': cons.get('converged', False)
                })
                
                if 'final_position_covariance' in cons:
                    cov = cons['final_position_covariance']
                    param_metrics.update({
                        'final_pos_cov_x': cov.get('x', 0),
                        'final_pos_cov_y': cov.get('y', 0),
                        'final_pos_cov_z': cov.get('z', 0),
                        'final_pos_cov_avg': (cov.get('x', 0) + cov.get('y', 0) + cov.get('z', 0)) / 3
                    })
            
            # 데이터 수집 통계
            if 'data_count' in data:
                count = data['data_count']
                param_metrics.update({
                    'pose_count': count.get('poses', 0),
                    'odom_count': count.get('odometry', 0),
                    'gps_fix_count': count.get('gps_fixes', 0),
                    'gps_vel_count': count.get('gps_velocities', 0)
                })
            
            # 원본 파라미터 파일 경로
            param_metrics['param_file'] = data.get('param_file', '')
            
            metrics[param_set] = param_metrics
        
        self.report_data['metrics'] = metrics
        return metrics
    
    def calculate_scores(self, metrics, weights=None):
        """각 파라미터 세트에 대한 점수 계산
        
        Args:
            metrics: 파라미터 세트별 메트릭 딕셔너리
            weights: 메트릭별 가중치 (딕셔너리)
        
        Returns:
            dict: 파라미터 세트별 점수
        """
        if not metrics:
            return {}
        
        # 기본 가중치 (낮을수록 좋음)
        default_weights = {
            'roll_std_deg': 1.0,      # 작을수록 좋음
            'pitch_std_deg': 1.0,     # 작을수록 좋음
            'yaw_std_deg': 1.0,       # 작을수록 좋음
            'speed_std': 1.0,         # 작을수록 좋음
            'final_pos_cov_avg': 2.0, # 작을수록 좋음
        }
        
        if weights:
            default_weights.update(weights)
        
        scores = {}
        
        for param_set, param_metrics in metrics.items():
            score = 0
            
            for metric, weight in default_weights.items():
                if metric in param_metrics:
                    # 기본적으로 낮은 값이 좋은 메트릭
                    score += param_metrics[metric] * weight
            
            scores[param_set] = score
        
        return scores
    
    def rank_parameters(self, metrics):
        """다양한 기준으로 파라미터 세트 순위 매기기"""
        if not metrics:
            return {}
        
        rankings = {}
        
        # 1. 종합 점수 (낮을수록 좋음)
        scores = self.calculate_scores(metrics)
        sorted_by_score = sorted(scores.items(), key=lambda x: x[1])
        rankings['overall'] = [param for param, _ in sorted_by_score]
        
        # 2. 자세 안정성
        if all('roll_std_deg' in metrics[param] for param in metrics):
            attitude_scores = {}
            for param, param_metrics in metrics.items():
                score = param_metrics['roll_std_deg'] + param_metrics['pitch_std_deg'] + param_metrics['yaw_std_deg']
                attitude_scores[param] = score
            
            sorted_by_attitude = sorted(attitude_scores.items(), key=lambda x: x[1])
            rankings['attitude_stability'] = [param for param, _ in sorted_by_attitude]
        
        # 3. 위치 정확도
        if all('final_pos_cov_avg' in metrics[param] for param in metrics):
            position_scores = {}
            for param, param_metrics in metrics.items():
                score = param_metrics['final_pos_cov_avg']
                position_scores[param] = score
            
            sorted_by_position = sorted(position_scores.items(), key=lambda x: x[1])
            rankings['position_accuracy'] = [param for param, _ in sorted_by_position]
        
        # 4. 속도 안정성
        if all('speed_std' in metrics[param] for param in metrics):
            velocity_scores = {}
            for param, param_metrics in metrics.items():
                score = param_metrics['speed_std']
                velocity_scores[param] = score
            
            sorted_by_velocity = sorted(velocity_scores.items(), key=lambda x: x[1])
            rankings['velocity_stability'] = [param for param, _ in sorted_by_velocity]
        
        self.report_data['rankings'] = rankings
        return rankings
    
    def visualize_rankings(self, metrics, rankings):
        """순위 시각화"""
        if not rankings:
            return {}
        
        plots = {}
        
        # 1. 종합 순위 시각화
        if 'overall' in rankings and len(rankings['overall']) > 0:
            plt.figure(figsize=(10, 6))
            
            params = rankings['overall']
            scores = [self.calculate_scores(metrics)[param] for param in params]
            
            plt.bar(params, scores)
            plt.title('파라미터 세트 종합 점수 (낮을수록 좋음)')
            plt.xlabel('파라미터 세트')
            plt.ylabel('점수')
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # 저장
            plot_file = Path(self.output_dir) / 'overall_ranking.png'
            plt.savefig(str(plot_file))
            plt.close()
            
            plots['overall_ranking'] = plot_file
        
        # 2. 메트릭별 비교 시각화
        if len(metrics) > 1:
            important_metrics = [
                ('자세 안정성', ['roll_std_deg', 'pitch_std_deg', 'yaw_std_deg']),
                ('위치 정확도', ['final_pos_cov_x', 'final_pos_cov_y']),
                ('속도 안정성', ['mean_speed', 'speed_std'])
            ]
            
            for title, metric_keys in important_metrics:
                if all(any(key in metrics[param] for key in metric_keys) for param in metrics):
                    plt.figure(figsize=(12, 6))
                    
                    param_sets = list(metrics.keys())
                    x = np.arange(len(param_sets))
                    width = 0.8 / len(metric_keys)
                    
                    for i, key in enumerate(metric_keys):
                        if all(key in metrics[param] for param in param_sets):
                            values = [metrics[param][key] for param in param_sets]
                            plt.bar(x + (i - len(metric_keys)/2 + 0.5) * width, values, width, label=key)
                    
                    plt.title(f'{title} 비교')
                    plt.xlabel('파라미터 세트')
                    plt.xticks(x, param_sets, rotation=45)
                    plt.legend()
                    plt.grid(True, axis='y')
                    
                    if 'cov' in title.lower():
                        plt.yscale('log')
                    
                    plt.tight_layout()
                    
                    # 저장
                    plot_file = os.path.join(self.output_dir, f'{title.replace(" ", "_")}_comparison.png')
                    plt.savefig(str(plot_file))
                    plt.close()
                    
                    plots[f'{title.replace(" ", "_")}_comparison'] = plot_file
        
        self.report_data['plots'] = plots
        return plots
    
    def select_best_parameters(self, rankings, metrics):
        """최적의 파라미터 세트 선택"""
        best_params = {}
        
        if not rankings or 'overall' not in rankings or not rankings['overall']:
            return best_params
        
        # 종합 점수 기준 최고 세트
        best_overall = rankings['overall'][0]
        best_params['overall'] = best_overall
        
        # 각 카테고리별 최고 세트 (다른 경우만)
        for category, ranked_params in rankings.items():
            if category != 'overall' and ranked_params and ranked_params[0] != best_overall:
                best_params[category] = ranked_params[0]
        
        # 추출 및 복사
        param_files = {
            category: metrics[param_set]['param_file']
            for category, param_set in best_params.items()
            if 'param_file' in metrics[param_set]
        }
        
        # 최적 파라미터 파일 복사
        for category, src_file in param_files.items():
            if src_file.exists():
                dest_file = Path(self.output_dir) / f'best_params_{category}.yaml'
                
                # 파일 복사
                import shutil
                shutil.copy2(src_file, dest_file)
                
                best_params[f'{category}_file'] = dest_file
                print(f"최적 파라미터 세트({category})가 {dest_file}에 저장되었습니다.")
        
        self.report_data['best_params'] = best_params
        return best_params
    
    def generate_report(self):
        """분석 결과 보고서 생성"""
        # 메트릭 데이터를 데이터프레임으로 변환
        if self.report_data['metrics']:
            metrics_df = pd.DataFrame.from_dict(self.report_data['metrics'], orient='index')
            
            # 보고서 파일 생성
            report_file = Path(self.output_dir) / 'analysis_report.html'
            
            with open(report_file, 'w') as f:
                f.write("<html>\n<head>\n")
                f.write("<title>EKF 파라미터 테스트 분석 보고서</title>\n")
                f.write("<style>\n")
                f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
                f.write("h1, h2, h3 { color: #333; }\n")
                f.write("table { border-collapse: collapse; width: 100%; }\n")
                f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
                f.write("th { background-color: #f2f2f2; }\n")
                f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
                f.write("img { max-width: 100%; height: auto; }\n")
                f.write(".best { font-weight: bold; color: green; }\n")
                f.write("</style>\n")
                f.write("</head>\n<body>\n")
                
                # 제목 및 요약
                f.write("<h1>EKF 파라미터 테스트 분석 보고서</h1>\n")
                f.write(f"<p>분석 시간: {self.report_data['analysis_time']}</p>\n")
                f.write(f"<p>테스트 결과 파일: {self.report_data['result_file']}</p>\n")
                
                # 최적 파라미터 요약
                f.write("<h2>최적 파라미터 세트</h2>\n")
                if 'best_params' in self.report_data and self.report_data['best_params']:
                    f.write("<ul>\n")
                    for category, param_set in self.report_data['best_params'].items():
                        if not category.endswith('_file'):
                            file_key = f'{category}_file'
                            file_path = self.report_data['best_params'].get(file_key, "")
                            file_name = os.path.basename(file_path) if file_path else ""
                            
                            f.write(f"<li><strong>{category}:</strong> {param_set}")
                            if file_name:
                                f.write(f" (파일: {file_name})")
                            f.write("</li>\n")
                    f.write("</ul>\n")
                else:
                    f.write("<p>최적 파라미터 세트를 결정할 수 없습니다.</p>\n")
                
                # 순위 표시
                f.write("<h2>파라미터 세트 순위</h2>\n")
                if 'rankings' in self.report_data and self.report_data['rankings']:
                    for category, ranked_params in self.report_data['rankings'].items():
                        f.write(f"<h3>{category} 기준</h3>\n")
                        f.write("<ol>\n")
                        for param in ranked_params:
                            f.write(f"<li>{param}</li>\n")
                        f.write("</ol>\n")
                
                # 메트릭 테이블
                f.write("<h2>성능 메트릭</h2>\n")
                if not metrics_df.empty:
                    metrics_html = metrics_df.to_html()
                    f.write(metrics_html)
                
                # 시각화 그래프
                f.write("<h2>시각화</h2>\n")
                if 'plots' in self.report_data and self.report_data['plots']:
                    for title, plot_file in self.report_data['plots'].items():
                        plot_title = title.replace('_', ' ').title()
                        f.write(f"<h3>{plot_title}</h3>\n")
                        f.write(f"<img src='{os.path.basename(plot_file)}' alt='{plot_title}'>\n")
                
                f.write("</body>\n</html>")
            
            print(f"분석 보고서가 {report_file}에 저장되었습니다.")
            return report_file
        else:
            print("메트릭 데이터가 없어 보고서를 생성할 수 없습니다.")
            return None
    
    def analyze(self):
        """전체 분석 프로세스 수행"""
        # 메트릭 추출
        metrics = self.extract_metrics()
        if not metrics:
            print("분석할 메트릭 데이터가 없습니다.")
            return False
        
        # 순위 매기기
        rankings = self.rank_parameters(metrics)
        
        # 시각화
        self.visualize_rankings(metrics, rankings)
        
        # 최적 파라미터 선택
        best_params = self.select_best_parameters(rankings, metrics)
        
        # 보고서 생성
        report_file = self.generate_report()
        
        return True


def find_result_files(directory, pattern='*test_results.json'):
    """디렉토리에서 패턴과 일치하는 결과 파일 찾기"""
    return glob.glob(os.path.join(directory, pattern))


def main(args=None):
    parser = argparse.ArgumentParser(description='EKF 파라미터 테스트 결과 분석 도구')
    parser.add_argument('-r', '--result-file', type=str,
                        help='분석할 테스트 결과 JSON 파일 경로')
    parser.add_argument('-d', '--result-dir', type=str,
                        help='테스트 결과 파일을 찾을 디렉토리')
    parser.add_argument('-o', '--output-dir', type=str,
                        help='분석 결과 저장 디렉토리 (기본값: 결과 파일과 동일 디렉토리)')
    
    parsed_args = parser.parse_args(args=args)
    
    # 결과 파일 결정
    result_file = None
    
    if parsed_args.result_file:
        result_file = parsed_args.result_file
    elif parsed_args.result_dir:
        result_files = find_result_files(parsed_args.result_dir)
        if result_files:
            # 가장 최근 파일 선택
            result_file = max(result_files, key=os.path.getmtime)
    
    if not result_file or not result_file.exists():
        print("오류: 분석할 테스트 결과 파일이 없습니다.")
        return 1
    
    print(f"EKF 파라미터 테스트 결과 분석 시작:")
    print(f"- 테스트 결과 파일: {result_file}")
    
    # 출력 디렉토리
    output_dir = parsed_args.output_dir
    
    # 분석 수행
    analyzer = TestResultAnalyzer(result_file, output_dir)
    success = analyzer.analyze()
    
    if success:
        print("분석이 성공적으로 완료되었습니다.")
        return 0
    else:
        print("분석 중 오류가 발생했습니다.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
