# GPS Visualization Tools

## Overview
GPS 경로 데이터를 RViz2에서 시각화하고 AerialMap과 정렬하는 도구들입니다.

## 구성 요소

### 1. gps_vis_node.py
- CSV 파일에서 GPS 경로 데이터를 읽어 RViz Path/Marker로 발행
- 건국대 일감호 (37.540091°N, 127.076555°E)를 원점(0,0)으로 설정
- UTM 좌표계 변환 사용

### 2. gps_to_cartesian.py
- ROS2 bag의 GPS 데이터를 로컬 Cartesian 좌표로 변환
- `/ublox_gps_node/fix` → `/gps/odometry` 변환
- robot_localization EKF와 통합 가능

### 3. fixed_gps_publisher.py
- AerialMap을 고정 위치에 표시하기 위한 더미 GPS 퍼블리셔
- `/aerialmap/fix` 토픽으로 고정 좌표 발행
- bag 파일 변경에 관계없이 지도 위치 고정

## 사용 방법

### AerialMap과 GPS 경로 정렬하기

```bash
# 터미널 1: 고정 GPS 퍼블리셔 (AerialMap용)
ros2 run cone_stellation fixed_gps_publisher.py

# 터미널 2: GPS 경로 시각화
ros2 run cone_stellation gps_vis_node.py

# 터미널 3: GPS to Cartesian 변환 (bag 재생 시)
ros2 run cone_stellation gps_to_cartesian.py

# 터미널 4: ROS2 bag 재생
ros2 bag play your_gps_data.bag
```

### RViz2 설정
1. AerialMap 디스플레이 추가
   - Topic: `/aerialmap/fix` (고정 위치용)
   - Zoom: 18-19 권장
   
2. Path 디스플레이 추가 (각 코스별)
   - Topics: `/gps_vis/course1`, `/gps_vis/course2`, ...
   
3. MarkerArray 디스플레이 추가
   - Topics: `/gps_vis/course1_markers`, ...

## 좌표계 정보
- Reference Point: 건국대 일감호 (37.540091°N, 127.076555°E)
- UTM Zone: 52N
- 로컬 좌표계: ENU (East-North-Up)

## 데이터 형식
CSV 파일은 다음 형식을 따라야 합니다:
```csv
Lat,Long
37.540091,127.076555
37.540120,127.076600
...
```