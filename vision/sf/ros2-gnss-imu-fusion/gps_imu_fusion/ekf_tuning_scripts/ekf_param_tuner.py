#!/usr/bin/env python3
# ekf_param_tuner.py
# EKF 파라미터 튜닝 프로세스를 총괄하는 마스터 스크립트

import argparse
import subprocess
import os
import sys
import yaml
import time
import shutil
from datetime import datetime
from pathlib import Path
from colorama import init, Fore, Style


def print_header(text):
    """컬러 헤더 출력"""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f" {text}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def print_success(text):
    """성공 메시지 출력"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_warning(text):
    """경고 메시지 출력"""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def print_error(text):
    """오류 메시지 출력"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_step(step_num, total_steps, text):
    """단계 메시지 출력"""
    print(f"{Fore.BLUE}[{step_num}/{total_steps}] {text}{Style.RESET_ALL}")


def run_command(cmd, desc, verbose=False, check=True):
    """명령 실행 및 결과 처리"""
    print(f"실행 중: {desc}...")
    
    if verbose:
        print(f"명령: {' '.join(cmd)}")
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if check and result.returncode != 0:
        if not verbose:
            print(f"표준 출력:\n{result.stdout.decode('utf-8')}")
            print(f"오류 출력:\n{result.stderr.decode('utf-8')}")
        print_error(f"{desc} 실패 (코드: {result.returncode})")
        return False
    
    print_success(f"{desc} 완료")
    return True


def setup_workspace(args):
    """작업 디렉토리 설정"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 기본 결과 디렉토리 설정
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(f"ekf_tuning_{timestamp}")
    
    # 디렉토리 생성
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 하위 디렉토리 생성
    static_results_dir = base_dir / "static_analysis"
    dynamic_results_dir = base_dir / "dynamic_tests"
    
    static_results_dir.mkdir(parents=True, exist_ok=True)
    dynamic_results_dir.mkdir(parents=True, exist_ok=True)
    
    print_success(f"작업 디렉토리 생성: {base_dir}")
    
    return {
        "base_dir": str(base_dir),
        "static_dir": str(static_results_dir),
        "dynamic_dir": str(dynamic_results_dir)
    }


def analyze_static_data(workspace, args):
    """정적 데이터 분석 (IMU 노이즈, GNSS 정확도, IMU 바이어스)"""
    print_header("정적 데이터 분석 시작")
    
    success = True
    static_dir = workspace["static_dir"]
    
    # 1. IMU 노이즈 분석
    print_step(1, 3, "IMU 노이즈 분석")
    
    imu_noise_cmd = [
        "python3", 
        "analyze_imu_noise.py",
        "-d", str(args.static_duration),
        "-o", static_dir
    ]
    
    if not run_command(imu_noise_cmd, "IMU 노이즈 분석", args.verbose):
        print_warning("IMU 노이즈 분석이 실패했지만 계속 진행합니다.")
        success = False
    
    # 2. GNSS 정확도 분석
    print_step(2, 3, "GNSS 정확도 분석")
    
    gnss_cmd = [
        "python3", 
        "analyze_gnss_accuracy.py",
        "-d", str(args.static_duration),
        "-o", static_dir
    ]
    
    if not run_command(gnss_cmd, "GNSS 정확도 분석", args.verbose):
        print_warning("GNSS 정확도 분석이 실패했지만 계속 진행합니다.")
        success = False
    
    # 3. IMU 바이어스 안정성 분석
    print_step(3, 3, "IMU 바이어스 안정성 분석")
    
    bias_cmd = [
        "python3", 
        "analyze_imu_bias.py",
        "-d", str(args.bias_duration),
        "-o", static_dir
    ]
    
    if not run_command(bias_cmd, "IMU 바이어스 안정성 분석", args.verbose):
        print_warning("IMU 바이어스 안정성 분석이 실패했지만 계속 진행합니다.")
        success = False
    
    return success


def generate_parameters(workspace, args):
    """정적 분석 결과에서 EKF 파라미터 생성"""
    print_header("EKF 파라미터 생성")
    
    static_dir = workspace["static_dir"]
    
    # 기본 파라미터 생성
    print_step(1, 3, "기본 EKF 파라미터 세트 생성")
    
    gen_cmd = [
        "python3", 
        "generate_ekf_params.py",
        "-i", static_dir,
        "-o", static_dir,
        "-f", "ekf_fusion_params_base.yaml"
    ]
    
    if not run_command(gen_cmd, "EKF 파라미터 생성", args.verbose):
        print_error("EKF 파라미터 생성 실패")
        return False
    
    base_param_file = Path(static_dir) / "ekf_fusion_params_base.yaml"
    
    if not base_param_file.exists():
        print_error(f"생성된 파라미터 파일을 찾을 수 없습니다: {base_param_file}")
        return False
    
    # 변형된 파라미터 세트 생성 (테스트용)
    print_step(2, 3, "변형된 EKF 파라미터 세트 생성")
    
    # 기본 파라미터 로드
    try:
        with open(str(base_param_file), 'r') as f:
            base_params = yaml.safe_load(f)
    except Exception as e:
        print_error(f"파라미터 파일 로드 실패: {e}")
        return False
    
    # 루트 노드 파라미터에 접근
    if 'ekf_fusion_node' in base_params and 'ros__parameters' in base_params['ekf_fusion_node']:
        params = base_params['ekf_fusion_node']['ros__parameters']
    else:
        print_error("파라미터 파일에서 필요한 키를 찾을 수 없습니다.")
        return False
    
    # 변형 파라미터 세트 생성
    variants = []
    
    # 변형 1: GPS 위치 노이즈 향상
    gps_precise = base_params.copy()
    if 'ekf_fusion_node' in gps_precise and 'ros__parameters' in gps_precise['ekf_fusion_node']:
        gps_precise['ekf_fusion_node']['ros__parameters']['gps_pos_noise_ne'] = params['gps_pos_noise_ne'] * 0.5
        gps_precise['ekf_fusion_node']['ros__parameters']['gps_pos_noise_d'] = params['gps_pos_noise_d'] * 0.5
        variants.append(("ekf_fusion_params_gps_precise.yaml", gps_precise, "GPS 정밀도 향상"))
    
    # 변형 2: IMU 노이즈 향상
    imu_precise = base_params.copy()
    if 'ekf_fusion_node' in imu_precise and 'ros__parameters' in imu_precise['ekf_fusion_node']:
        imu_precise['ekf_fusion_node']['ros__parameters']['accel_noise'] = params['accel_noise'] * 0.5
        imu_precise['ekf_fusion_node']['ros__parameters']['gyro_noise'] = params['gyro_noise'] * 0.5
        variants.append(("ekf_fusion_params_imu_precise.yaml", imu_precise, "IMU 정밀도 향상"))
    
    # 변형 3: GPS 속도 노이즈 향상
    vel_precise = base_params.copy()
    if 'ekf_fusion_node' in vel_precise and 'ros__parameters' in vel_precise['ekf_fusion_node']:
        vel_precise['ekf_fusion_node']['ros__parameters']['gps_vel_noise_ne'] = params['gps_vel_noise_ne'] * 0.5
        vel_precise['ekf_fusion_node']['ros__parameters']['gps_vel_noise_d'] = params['gps_vel_noise_d'] * 0.5
        variants.append(("ekf_fusion_params_vel_precise.yaml", vel_precise, "속도 정밀도 향상"))
    
    # 변형 4: 바이어스 안정성 향상
    bias_stable = base_params.copy()
    if 'ekf_fusion_node' in bias_stable and 'ros__parameters' in bias_stable['ekf_fusion_node']:
        bias_stable['ekf_fusion_node']['ros__parameters']['accel_bias_tau'] = params['accel_bias_tau'] * 2.0
        bias_stable['ekf_fusion_node']['ros__parameters']['gyro_bias_tau'] = params['gyro_bias_tau'] * 2.0
        variants.append(("ekf_fusion_params_bias_stable.yaml", bias_stable, "바이어스 안정성 향상"))
    
    # 변형 5: GPS 헤딩 최소 속도 최적화
    heading_opt = base_params.copy()
    if 'ekf_fusion_node' in heading_opt and 'ros__parameters' in heading_opt['ekf_fusion_node']:
        heading_opt['ekf_fusion_node']['ros__parameters']['min_speed_for_gnss_heading'] = params['min_speed_for_gnss_heading'] * 0.8
        variants.append(("ekf_fusion_params_heading_opt.yaml", heading_opt, "GPS 헤딩 최적화"))
    
    # 변형 파라미터 파일 저장
    for filename, params_data, desc in variants:
        filepath = Path(static_dir) / filename
        try:
            with open(str(filepath), 'w') as f:
                # 주석 추가
                f.write(f"# EKF 퓨전 노드 파라미터 - {desc}\n")
                f.write(f"# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# 이 파일은 자동으로 생성되었습니다.\n\n")
                
                yaml.dump(params_data, f, default_flow_style=False)
            print_success(f"파라미터 세트 생성: {filename} ({desc})")
        except Exception as e:
            print_error(f"파라미터 파일 저장 실패: {e}")
    
    print_step(3, 3, "파라미터 세트 생성 완료")
    print_success(f"총 {len(variants) + 1}개 파라미터 세트가 {static_dir}에 생성되었습니다.")
    
    return True


def test_parameters(workspace, args):
    """생성된 파라미터 세트 테스트"""
    print_header("파라미터 세트 테스트 (동적 데이터)")
    
    if not args.test_params:
        print_warning("파라미터 테스트를 건너뜁니다.")
        return True
    
    static_dir = workspace["static_dir"]
    dynamic_dir = workspace["dynamic_dir"]
    
    # 테스트할 파라미터 파일 찾기
    param_files = [str(Path(static_dir) / f) for f in os.listdir(static_dir) 
                  if f.startswith("ekf_fusion_params") and f.endswith(".yaml")]
    
    if not param_files:
        print_error("테스트할 파라미터 파일이 없습니다.")
        return False
    
    # 파라미터 테스트 실행
    print_step(1, 2, f"{len(param_files)}개 파라미터 세트 테스트")
    
    test_cmd = [
        "python3", 
        "test_ekf_params.py",
        "-p"] + param_files + [
        "-t", str(args.test_duration),
        "-o", dynamic_dir,
        "-n", "dynamic_test"
    ]
    
    if not run_command(test_cmd, "파라미터 테스트", args.verbose):
        print_error("파라미터 테스트 실패")
        return False
    
    # 결과 분석
    print_step(2, 2, "테스트 결과 분석")
    
    result_file = Path(dynamic_dir) / "dynamic_test_test_results.json"
    
    if not result_file.exists():
        print_error(f"테스트 결과 파일을 찾을 수 없습니다: {result_file}")
        return False
    
    analyze_cmd = [
        "python3", 
        "analyze_test_results.py",
        "-r", result_file,
        "-o", dynamic_dir
    ]
    
    if not run_command(analyze_cmd, "테스트 결과 분석", args.verbose):
        print_error("테스트 결과 분석 실패")
        return False
    
    return True


def finalize_parameters(workspace, args):
    """최종 파라미터 선택 및 저장"""
    print_header("최종 파라미터 선택")
    
    base_dir = workspace["base_dir"]
    static_dir = workspace["static_dir"]
    dynamic_dir = workspace["dynamic_dir"]
    
    # 최적 파라미터 파일 결정
    optimal_param_file = None
    
    if args.test_params:
        # 동적 테스트 결과에서 최적 파라미터 찾기
        best_file = Path(dynamic_dir) / "best_params_overall.yaml"
        if best_file.exists():
            optimal_param_file = str(best_file)
            print_success(f"동적 테스트 결과에서 최적 파라미터 찾음: {best_file.name}")
    
    if not optimal_param_file:
        # 동적 테스트를 수행하지 않았거나 결과가 없는 경우 기본 파라미터 사용
        base_param_file = Path(static_dir) / "ekf_fusion_params_base.yaml"
        if base_param_file.exists():
            optimal_param_file = str(base_param_file)
            print_success(f"정적 분석 기본 파라미터 사용: {base_param_file.name}")
    
    if not optimal_param_file:
        print_error("최적 파라미터 파일을 찾을 수 없습니다.")
        return False
    
    # 최종 파라미터 파일 생성
    final_param_file = Path(base_dir) / "ekf_fusion_params_final.yaml"
    
    try:
        shutil.copy2(optimal_param_file, str(final_param_file))
        print_success(f"최종 파라미터 파일 생성: {final_param_file}")
    except Exception as e:
        print_error(f"최종 파라미터 파일 생성 실패: {e}")
        return False
    
    # 요약 정보 생성
    summary_file = Path(base_dir) / "tuning_summary.yaml"
    
    summary = {
        'tuning_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'workspace': base_dir,
        'static_analysis_dir': static_dir,
        'dynamic_test_dir': dynamic_dir if args.test_params else "테스트하지 않음",
        'final_parameters': str(final_param_file),
        'optimal_source': Path(optimal_param_file).name if optimal_param_file else None
    }
    
    try:
        with open(str(summary_file), 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        print_success(f"튜닝 요약 정보 저장: {summary_file}")
    except Exception as e:
        print_error(f"요약 정보 저장 실패: {e}")
    
    return True


def run_tuning_process(args):
    """전체 튜닝 프로세스 실행"""
    # 컬러 출력 초기화
    init()
    
    print_header("EKF 파라미터 튜닝 프로세스 시작")
    
    # 1. 작업 디렉토리 설정
    workspace = setup_workspace(args)
    
    # 2. 정적 데이터 분석
    static_success = analyze_static_data(workspace, args)
    
    # 3. 기본 파라미터 생성
    params_success = generate_parameters(workspace, args)
    if not params_success:
        print_error("파라미터 생성 실패로 프로세스를 중단합니다.")
        return False
    
    # 4. 파라미터 테스트 (선택사항)
    if args.test_params:
        test_success = test_parameters(workspace, args)
    else:
        print_warning("파라미터 테스트를 건너뜁니다.")
        test_success = True
    
    # 5. 최종 파라미터 선택
    final_success = finalize_parameters(workspace, args)
    
    # 최종 결과 출력
    if static_success and params_success and test_success and final_success:
        print_header("EKF 파라미터 튜닝 프로세스 완료")
        print_success(f"최종 파라미터 파일: {Path(workspace['base_dir']) / 'ekf_fusion_params_final.yaml'}")
        print_success(f"모든 결과는 '{workspace['base_dir']}' 디렉토리에 저장되었습니다.")
        return True
    else:
        print_header("EKF 파라미터 튜닝 프로세스 완료 (일부 단계 실패)")
        if not static_success:
            print_warning("- 정적 데이터 분석에 일부 실패했습니다.")
        if not params_success:
            print_error("- 파라미터 생성에 실패했습니다.")
        if not test_success:
            print_error("- 파라미터 테스트에 실패했습니다.")
        if not final_success:
            print_error("- 최종 파라미터 선택에 실패했습니다.")
        
        print_warning(f"결과는 '{workspace['base_dir']}' 디렉토리에서 확인하세요.")
        return False


def main():
    parser = argparse.ArgumentParser(description='EKF 파라미터 자동 튜닝 도구')
    
    # 기본 옵션
    parser.add_argument('-o', '--output-dir', type=str,
                        help='결과 저장 디렉토리 (기본값: 현재 디렉토리에 타임스탬프 폴더)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='자세한 출력 표시')
    
    # 정적 데이터 분석 옵션
    parser.add_argument('--static-duration', type=float, default=30.0,
                        help='정적 데이터 분석 시간(초), 기본값: 30.0')
    parser.add_argument('--bias-duration', type=float, default=300.0,
                        help='바이어스 안정성 분석 시간(초), 기본값: 300.0')
    
    # 파라미터 테스트 옵션
    parser.add_argument('--test-params', action='store_true',
                        help='파라미터 세트 테스트 수행 (동적 데이터 필요)')
    parser.add_argument('--test-duration', type=float, default=60.0,
                        help='파라미터 테스트 시간(초), 기본값: 60.0')
    
    args = parser.parse_args()
    
    try:
        success = run_tuning_process(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        print_warning("\n사용자가 프로세스를 중단했습니다.")
        return 1
    except Exception as e:
        print_error(f"\n예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
