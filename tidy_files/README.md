# 아쿠아포닉스 모니터링 시스템 v2.0
Edge-Server 분리 아키텍처 기반의 경량화된 아쿠아포닉스 통합 모니터링 시스템

### 주요 특징
- Edge-Server 분리: 실시간 제어(Edge) + 고정밀 분석(Server)
- 경량화 설계: 저사양 하드웨어 최적화 (라즈베리파이 + 일반 서버)
- 통합 설정: 단일 config.yaml로 모든 설정 관리
- 확장 가능: 센서/제어기 추가 용이
- 실시간 대시보드: 웹 기반 모니터링 및 제어

### 📁 프로젝트 구조
```
aquaponics_v2/
├── 📄 config.yaml              # 통합 설정 파일
├── 📄 requirements.txt         # Python 의존성
├── 📄 README.md               # 이 파일
├── 
├── 📂 common/                 # 공통 모듈
│   ├── data_types.py          # 데이터 구조 정의
│   ├── mqtt_client.py         # MQTT 클라이언트
│   ├── yolo_detector.py       # YOLO 탐지기
│   └── sensor_manager.py      # 센서 관리자
│
├── 📂 edge/                   # Edge 디바이스 (라즈베리파이)
│   ├── edge_controller.py     # 메인 컨트롤러
│   ├── control_policy.py      # 제어 정책
│   └── camera_manager.py      # 카메라 관리자
│
├── 📂 server/                 # 서버 (우분투)
│   ├── lightweight_server.py  # 경량 서버 (메인)
│   ├── database_manager.py    # 데이터베이스 관리
│   ├── analysis_engine.py     # 분석 엔진
│   └── alert_manager.py       # 알림 관리자
│
├── 📂 dashboard/              # 웹 대시보드
│   └── simple_dashboard.py    # Streamlit 대시보드
│
├── 📂 scripts/                # 유틸리티 스크립트
│   ├── start_edge.sh          # Edge 시작
│   ├── start_server.sh        # 서버 시작
│   ├── start_dashboard.sh     # 대시보드 시작
│   ├── stop_all.sh           # 전체 중지
│   ├── install_dependencies.sh # 의존성 설치
│   └── system_check.sh       # 시스템 체크
│
├── 📂 models/                 # YOLO 모델들
│   ├── yolov5n.pt            # Edge용 경량 모델
│   └── yolov8m.pt            # Server용 정밀 모델
│
├── 📂 logs/                   # 로그 파일들
├── 📂 data/                   # 데이터 저장소
└── 📂 debug/                  # 디버그 파일들
```

### 빠른 시작
#### 1. 시스템 요구사항
Edge Device (라즈베리파이 4 권장)
- RAM: 4GB 이상
- 저장소: 32GB+ MicroSD
- 카메라: USB 또는 라즈베리파이 카메라
- Python: 3.8+

Server (저사양 우분투 서버)
- CPU: 2코어 이상
- RAM: 4GB 이상 (권장 8GB)
- 저장소: 50GB+
- Python: 3.8+

#### 2. 설치
```
# 저장소 클론
git clone <repository-url>
cd aquaponics_v2

# 의존성 자동 설치
chmod +x scripts/*.sh
./scripts/install_dependencies.sh

# 설정 파일 복사 및 수정
cp config.yaml.example config.yaml
vi config.yaml  # 환경에 맞게 수정
```

#### 3. 실행
# 서버 시작 (우분투 서버에서)
```
./scripts/start_server.sh

# Edge 디바이스 시작 (라즈베리파이에서)
./scripts/start_edge.sh

# 대시보드 시작 (별도 컴퓨터에서)
./scripts/start_dashboard.sh
```

#### 4. 접속
- 대시보드: http://localhost:8080
- 서버 API: http://서버IP:5000
- MQTT: 서버IP:1883

### 설정
config.yaml 주요 설정
```
# Edge 디바이스
edge:
  device_id: "edge_001"
  camera:
    rtsp_url: "rtsp://카메라IP:554/stream"
  yolo:
    model_path: "models/yolov5n.pt"

# 서버
server:
  host: "서버IP"
  port: 5000
  yolo:
    model_path: "models/yolov8m.pt"

# MQTT
mqtt:
  broker_host: "서버IP"
  broker_port: 1883

# 센서 (GPIO 핀 번호)
sensors:
  water:
    temperature: {pin: 18}
    ph: {pin: 19}
```

### 시스템 관리
상태 확인
```
./scripts/system_check.sh
```

로그 확인 
```
tail -f logs/edge_edge_001.log    # Edge 로그
tail -f logs/server_server_001.log  # 서버 로그
```

서비스 중지
```
./scripts/stop_all.sh
```

재시작
```
./scripts/stop_all.sh
sleep 5
./scripts/start_server.sh &    # 서버
./scripts/start_edge.sh &      # Edge
./scripts/start_dashboard.sh & # 대시보드
```

### 하드웨어 연결
라즈베리파이 GPIO 핀맵
```
센서 연결:
- 수온 센서 (DS18B20): GPIO 18
- pH 센서: GPIO 19 (ADC 필요)
- EC 센서: GPIO 20 (ADC 필요)  
- 용존산소: GPIO 21 (ADC 필요)
- 대기온도/습도: GPIO 22 (DHT22)
- 조도센서: GPIO 24 (Analog)

제어 장치 연결:
- 워터펌프: GPIO 25 (릴레이)
- 히터: GPIO 26 (PWM)
- pH 도징펌프: GPIO 27 (릴레이)
- 사료급이기: GPIO 28 (서보)
- LED 조명: GPIO 29 (PWM)
```


### API 엔드포인트
서버 API
```
GET  /api/system/status           # 시스템 상태
GET  /api/environment/trends      # 환경 트렌드
GET  /api/objects/tracking        # 객체 추적
GET  /api/alerts/recent          # 최근 알림
POST /api/control/command        # 제어 명령
POST /api/edge/events            # Edge 이벤트 수신
```

사용 예시
```
# 시스템 상태 조회
curl http://서버IP:5000/api/system/status

# 제어 명령 전송
curl -X POST http://서버IP:5000/api/control/command \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "water_pump",
    "command_type": "pwm", 
    "target_value": 0.8,
    "duration": 300
  }'
```

### 문제 해결
일반적인 문제들
1. 카메라 연결 실패
```
# USB 카메라 확인
lsusb
v4l2-ctl --list-devices

# 라즈베리파이 카메라 확인
vcgencmd get_camera
```

2. MQTT 연결 실패
```
# MQTT 브로커 상태 확인
sudo systemctl status mosquitto

# 수동 테스트
mosquitto_pub -h 서버IP -t "test" -m "hello"
mosquitto_sub -h 서버IP -t "test"
```

3. 모델 로드 실패
```
# 모델 파일 확인
ls -la models/
file models/yolov5n.pt

# 다시 다운로드
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt -O models/yolov5n.pt
```

4. 센서 읽기 실패
```
# GPIO 권한 확인
groups $USER
sudo usermod -a -G gpio $USER

# I2C 활성화 (라즈베리파이)
sudo raspi-config
# Interface Options > I2C > Enable
```

#### 로그 분석
```
# 오류 로그 필터링
grep -i error logs/*.log
grep -i warning logs/*.log

# 성능 확인
grep "fps\|processing_time" logs/edge_*.log
```

## 업데이트 및 백업
시스템 업데이트 
```
git pull origin main
./scripts/install_dependencies.sh
./scripts/stop_all.sh
./scripts/start_server.sh
```

데이터 백업
```
# 데이터베이스 백업
cp data/aquaponics.db backup/aquaponics_$(date +%Y%m%d).db

# 설정 백업
cp config.yaml backup/config_$(date +%Y%m%d).yaml

# 로그 아카이브
tar -czf backup/logs_$(date +%Y%m%d).tar.gz logs/
```


🤝 기여 및 개발
새로운 센서 추가

config.yaml에 센서 정의
`common/sensor_manager.py`에 읽기 함수 추가
`common/data_types.py`에 데이터 타입 확장

새로운 제어기 추가

config.yaml에 제어기 정의
`edge/control_policy.py`에 제어 로직 추가
하드웨어 연결

커스텀 분석 추가

`server/analysis_engine.py`에 분석 함수 추가
API 엔드포인트 추가 (필요시)
대시보드에 시각화 추가

📝 라이선스
이 프로젝트는 다음 오픈소스 라이브러리를 사용합니다:

YOLOv5/YOLOv8: GPL-3.0 License
OpenCV: Apache License 2.0
PyTorch: BSD License
Flask: BSD License
Streamlit: Apache License 2.0

🆘 지원

🐛 버그 리포트: GitHub Issues
💬 질문: GitHub Discussions
📚 문서: /docs 폴더

## 📈 버전 히스토리
### v2.0.0 (현재)
- Edge-Server 분리 아키텍처
- 경량화 최적화
- 통합 설정 시스템
- 실시간 대시보드

### v1.0.0
- 기본 모니터링 시스템
- 단일 서버 구조



최종 업데이트: 2025년 7월
메인테이너: sunjun Hwang

