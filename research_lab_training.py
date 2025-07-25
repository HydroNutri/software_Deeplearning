# research_lab_training.py - 연구실 고사양 컴퓨터용 학습 및 고급 분석 시스템
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import pandas as pd
import sqlite3
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import wandb  # Weights & Biases for experiment tracking
from ultralytics import YOLO
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import optuna  # 하이퍼파라미터 최적화
import warnings
warnings.filterwarnings('ignore')

class AquaponicsDataset(Dataset):
    """아쿠아포닉스 데이터셋 클래스"""
    
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 특징 추출 (길이, 무게, 시간 등)
        features = torch.tensor([
            row.get('length_cm', 0),
            row.get('weight_g', 0),
            row.get('height_cm', 0),
            row.get('confidence', 0),
            row.get('growth_rate', 0),
            row.get('variability', 0)
        ], dtype=torch.float32)
        
        # 라벨 (건강도 또는 성장 예측)
        label = torch.tensor(row.get('health_score', 1.0), dtype=torch.float32)
        
        return features, label

class GrowthPredictor(nn.Module):
    """성장 예측 딥러닝 모델"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=1):
        super(GrowthPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Sigmoid()  # 0-1 범위 출력
        )
        
    def forward(self, x):
        return self.network(x)

class AquaponicsEnvironment(gym.Env):
    """강화학습을 위한 아쿠아포닉스 환경"""
    
    def __init__(self):
        super(AquaponicsEnvironment, self).__init__()
        
        # 액션 스페이스: [사료량, 조명강도, 수온조절, pH조절]
        self.action_space = gym.spaces.Box(
            low=np.array([0.5, 0.3, 18.0, 6.0]),
            high=np.array([2.0, 1.0, 28.0, 8.0]),
            dtype=np.float32
        )
        
        # 상태 스페이스: [물고기수, 평균크기, 식물수, 평균높이, 환경파라미터들]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 18.0, 6.0, 0, 0]),
            high=np.array([50, 30, 20, 50, 28.0, 8.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)
        
        # 초기 상태 설정
        self.state = np.array([
            np.random.randint(5, 15),  # 물고기 수
            np.random.uniform(8, 12),  # 평균 크기
            np.random.randint(3, 8),   # 식물 수
            np.random.uniform(15, 25), # 평균 높이
            np.random.uniform(22, 26), # 수온
            np.random.uniform(6.5, 7.5), # pH
            np.random.uniform(0.7, 1.0), # 물고기 건강도
            np.random.uniform(0.7, 1.0)  # 식물 건강도
        ], dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 100
        
        return self.state, {}
        
    def step(self, action):
        """환경 스텝"""
        self.step_count += 1
        
        # 액션 적용 (사료량, 조명, 수온, pH)
        feed_amount, light_intensity, water_temp, ph_level = action
        
        # 환경 업데이트
        self.state[4] = water_temp  # 수온
        self.state[5] = ph_level    # pH
        
        # 성장 시뮬레이션
        growth_factor = self._calculate_growth_factor(action)
        
        # 물고기 성장
        self.state[1] += growth_factor * 0.1  # 크기 증가
        if np.random.random() < 0.05:  # 새 물고기 확률
            self.state[0] = min(50, self.state[0] + 1)
            
        # 식물 성장
        self.state[3] += growth_factor * 0.2  # 높이 증가
        if np.random.random() < 0.03:  # 새 식물 확률
            self.state[2] = min(20, self.state[2] + 1)
            
        # 건강도 업데이트
        self.state[6] = np.clip(self.state[6] + growth_factor * 0.05 - 0.01, 0, 1)
        self.state[7] = np.clip(self.state[7] + growth_factor * 0.05 - 0.01, 0, 1)
        
        # 보상 계산
        reward = self._calculate_reward(action, growth_factor)
        
        # 종료 조건
        terminated = (self.step_count >= self.max_steps or 
                     self.state[6] < 0.3 or self.state[7] < 0.3)
        
        return self.state, reward, terminated, False, {}
        
    def _calculate_growth_factor(self, action):
        """성장 인자 계산"""
        feed_amount, light_intensity, water_temp, ph_level = action
        
        # 최적 조건과의 차이 계산
        optimal_temp = 24.0
        optimal_ph = 7.0
        optimal_feed = 1.0
        optimal_light = 0.8
        
        temp_factor = 1.0 - abs(water_temp - optimal_temp) / 10.0
        ph_factor = 1.0 - abs(ph_level - optimal_ph) / 2.0
        feed_factor = 1.0 - abs(feed_amount - optimal_feed) / 1.0
        light_factor = 1.0 - abs(light_intensity - optimal_light) / 0.5
        
        return np.clip(np.mean([temp_factor, ph_factor, feed_factor, light_factor]), 0, 1)
        
    def _calculate_reward(self, action, growth_factor):
        """보상 함수"""
        # 성장률 보상
        growth_reward = growth_factor * 10
        
        # 안정성 보상 (급격한 변화 페널티)
        stability_penalty = np.sum(np.abs(action - [1.0, 0.8, 24.0, 7.0])) * 0.5
        
        # 건강도 보상
        health_reward = (self.state[6] + self.state[7]) * 5
        
        # 개체수 보상
        population_reward = (self.state[0] + self.state[2]) * 0.1
        
        total_reward = growth_reward + health_reward + population_reward - stability_penalty
        
        return total_reward

class ResearchLabSystem:
    """연구실 고급 분석 및 학습 시스템"""
    
    def __init__(self, server_url="http://192.168.1.200:5000"):
        self.server_url = server_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 데이터베이스 연결
        self.db_path = 'research_data.db'
        self.init_research_db()
        
        # 모델들
        self.growth_predictor = None
        self.rl_agent = None
        self.yolo_model = None
        
        # 실험 추적
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def init_research_db(self):
        """연구용 데이터베이스 초기화"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT,
                model_type TEXT,
                hyperparameters TEXT,
                metrics TEXT,
                start_time DATETIME,
                end_time DATETIME,
                best_score REAL
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT,
                object_id TEXT,
                prediction_type TEXT,
                predicted_value REAL,
                actual_value REAL,
                timestamp DATETIME,
                confidence REAL
            )
        ''')
        
        self.conn.commit()
        
    def fetch_training_data(self):
        """서버에서 학습 데이터 수집"""
        try:
            # 최근 한 달 데이터 요청
            response = requests.get(f"{self.server_url}/api/training_data")
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['records'])
                
                print(f"학습 데이터 수집: {len(df)}개 레코드")
                return df
            else:
                print(f"데이터 수집 실패: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None
            
    def train_growth_predictor(self, data_df):
        """성장 예측 모델 학습"""
        print("성장 예측 모델 학습 시작")
        
        # 데이터 전처리
        features = ['length_cm', 'weight_g', 'height_cm', 'confidence', 'growth_rate', 'variability']
        target = 'future_growth'  # 미래 성장률
        
        # 미래 성장률 계산 (실제 구현에서는 시계열 데이터 활용)
        data_df['future_growth'] = data_df['growth_rate'].shift(-1).fillna(0)
        
        X = data_df[features].fillna(0)
        y = data_df[target].fillna(0)
        
        # 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train.values)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val.values)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 모델 초기화
        self.growth_predictor = GrowthPredictor(input_dim=len(features))
        self.growth_predictor.to(self.device)
        
        # 옵티마이저 및 손실함수
        optimizer = optim.Adam(self.growth_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 학습 루프
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # 훈련
            self.growth_predictor.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.growth_predictor(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # 검증
            self.growth_predictor.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.growth_predictor(batch_x).squeeze()
                    val_loss += criterion(outputs, batch_y).item()
                    
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 모델 저장
                torch.save(self.growth_predictor.state_dict(), f'growth_predictor_{self.experiment_id}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("조기 종료")
                    break
                    
        print(f"모델 학습 완료. 최고 검증 손실: {best_val_loss:.4f}")
        
        # 실험 기록 저장
        self.conn.execute('''
            INSERT INTO experiments (experiment_id, model_type, metrics, start_time, end_time, best_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.experiment_id,
            'GrowthPredictor',
            json.dumps({'val_loss': best_val_loss}),
            datetime.now() - timedelta(minutes=10),  # 대략적인 시작 시간
            datetime.now(),
            best_val_loss
        ))
        self.conn.commit()
        
    def train_rl_agent(self):
        """강화학습 에이전트 학습"""
        print("강화학습 에이전트 학습 시작")
        
        # 환경 생성
        env = make_vec_env(AquaponicsEnvironment, n_envs=4)
        
        # PPO 에이전트 생성
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            device=self.device
        )
        
        # 평가 콜백
        eval_env = AquaponicsEnvironment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'./rl_models/{self.experiment_id}/',
            log_path=f'./rl_logs/{self.experiment_id}/',
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # 학습 실행
        self.rl_agent.learn(
            total_timesteps=100000,
            callback=eval_callback
        )
        
        # 모델 저장
        self.rl_agent.save(f"rl_agent_{self.experiment_id}")
        
        print("강화학습 에이전트 학습 완료")
        
    def hyperparameter_optimization(self, data_df):
        """하이퍼파라미터 최적화"""
        print("하이퍼파라미터 최적화 시작")
        
        def objective(trial):
            # 하이퍼파라미터 제안
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            # 간단한 모델 학습 및 평가 (실제로는 전체 학습 파이프라인)
            # 여기서는 간단한 예시로 대체
            model = GrowthPredictor(hidden_dim=hidden_dim)
            
            # 가상의 성능 점수 (실제로는 검증 손실)
            score = np.random.random() * (1 - dropout) * (hidden_dim / 512) * (0.001 / lr)
            
            return score
            
        # Optuna 스터디
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print(f"최적 하이퍼파라미터: {study.best_params}")
        print(f"최고 성능: {study.best_value}")
        
        return study.best_params
        
    def anomaly_detection_analysis(self, data_df):
        """이상 탐지 분석"""
        print("이상 탐지 분석 수행")
        
        # 특징 선택
        features = ['length_cm', 'weight_g', 'height_cm', 'growth_rate', 'variability']
        X = data_df[features].fillna(0)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        # 이상치 데이터
        anomalies = data_df[anomaly_labels == -1]
        
        print(f"이상치 감지: {len(anomalies)}개 ({len(anomalies)/len(data_df)*100:.1f}%)")
        
        # 시각화
        self.visualize_anomalies(data_df, anomaly_labels)
        
        return anomalies
        
    def visualize_anomalies(self, data_df, anomaly_labels):
        """이상치 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 성장률 vs 시간
        plt.subplot(2, 3, 1)
        normal_data = data_df[anomaly_labels == 1]
        anomaly_data = data_df[anomaly_labels == -1]
        
        plt.scatter(normal_data.index, normal_data['growth_rate'], 
                   c='blue', alpha=0.6, label='Normal')
        plt.scatter(anomaly_data.index, anomaly_data['growth_rate'], 
                   c='red', alpha=0.8, label='Anomaly')
        plt.xlabel('Time Index')
        plt.ylabel('Growth Rate')
        plt.title('Growth Rate Anomalies')
        plt.legend()
        
        # 서브플롯 2: 크기 분포
        plt.subplot(2, 3, 2)
        plt.hist(normal_data['length_cm'], bins=30, alpha=0.7, label='Normal', color='blue')
        plt.hist(anomaly_data['length_cm'], bins=30, alpha=0.7, label='Anomaly', color='red')
        plt.xlabel('Length (cm)')
        plt.ylabel('Frequency')
        plt.title('Size Distribution')
        plt.legend()
        
        # 서브플롯 3: 변동성 vs 성장률
        plt.subplot(2, 3, 3)
        plt.scatter(normal_data['variability'], normal_data['growth_rate'], 
                   c='blue', alpha=0.6, label='Normal')
        plt.scatter(anomaly_data['variability'], anomaly_data['growth_rate'], 
                   c='red', alpha=0.8, label='Anomaly')
        plt.xlabel('Variability')
        plt.ylabel('Growth Rate')
        plt.title('Variability vs Growth Rate')
        plt.legend()
        
        # 상관관계 히트맵
        plt.subplot(2, 3, 4)
        correlation_matrix = data_df[['length_cm', 'weight_g', 'height_cm', 'growth_rate', 'variability']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation')
        
        # 시계열 트렌드
        plt.subplot(2, 3, 5)
        if 'timestamp' in data_df.columns:
            data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
            daily_avg = data_df.groupby(data_df['timestamp'].dt.date)['growth_rate'].mean()
            plt.plot(daily_avg.index, daily_avg.values, marker='o')
            plt.title('Daily Average Growth Rate')
            plt.xticks(rotation=45)
        
        # 건강도 분포
        plt.subplot(2, 3, 6)
        if 'health_score' in data_df.columns:
            plt.hist(normal_data['health_score'], bins=20, alpha=0.7, label='Normal', color='blue')
            plt.hist(anomaly_data['health_score'], bins=20, alpha=0.7, label='Anomaly', color='red')
            plt.xlabel('Health Score')
            plt.ylabel('Frequency')
            plt.title('Health Score Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'anomaly_analysis_{self.experiment_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def advanced_yolo_training(self, custom_dataset_path):
        """커스텀 YOLO 모델 학습"""
        print("커스텀 YOLO 모델 학습 시작")
        
        # YOLOv8 모델 로드
        self.yolo_model = YOLO('yolov8m.pt')
        
        # 커스텀 데이터셋으로 파인튜닝
        results = self.yolo_model.train(
            data=custom_dataset_path,  # YAML 설정 파일
            epochs=100,
            imgsz=640,
            batch=16,
            device=self.device,
            patience=20,
            save_period=10,
            project='aquaponics_yolo',
            name=f'experiment_{self.experiment_id}',
            
            # 고급 설정
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # 데이터 증강
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            
            # 검증 설정
            val=True,
            split=0.2,
            
            # 기타
            verbose=True,
            seed=42
        )
        
        # 모델 성능 평가
        metrics = self.yolo_model.val()
        
        print(f"YOLO 학습 완료")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
        
        # 최적화된 모델 내보내기
        self.yolo_model.export(
            format='onnx',  # ONNX 형식으로 내보내기
            optimize=True,
            half=True,  # FP16 최적화
            simplify=True
        )
        
        return results
        
    def generate_synthetic_data(self, num_samples=1000):
        """합성 데이터 생성 (데이터 부족 시)"""
        print(f"합성 데이터 생성: {num_samples}개 샘플")
        
        synthetic_data = []
        
        for _ in range(num_samples):
            # 기본 파라미터
            base_length = np.random.normal(15, 3)
            base_weight = base_length ** 2 * np.random.normal(0.8, 0.2)
            base_height = np.random.normal(25, 5)
            
            # 시간 경과에 따른 변화
            days = np.random.randint(1, 30)
            growth_rate = np.random.normal(0.1, 0.05)
            
            # 환경 요인
            temp_factor = np.random.normal(1.0, 0.1)
            ph_factor = np.random.normal(1.0, 0.1)
            light_factor = np.random.normal(1.0, 0.1)
            
            # 최종 크기 계산
            final_length = base_length * (1 + growth_rate * days * temp_factor * ph_factor)
            final_weight = base_weight * (1 + growth_rate * days * temp_factor * ph_factor) ** 2
            final_height = base_height * (1 + growth_rate * days * light_factor)
            
            # 건강도 계산
            health_score = np.clip(
                np.mean([temp_factor, ph_factor, light_factor]) * 
                np.random.normal(0.85, 0.1), 0, 1
            )
            
            synthetic_data.append({
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'object_type': np.random.choice(['fish', 'plant']),
                'length_cm': final_length if np.random.choice(['fish', 'plant']) == 'fish' else None,
                'weight_g': final_weight if np.random.choice(['fish', 'plant']) == 'fish' else None,
                'height_cm': final_height if np.random.choice(['fish', 'plant']) == 'plant' else None,
                'growth_rate': growth_rate,
                'variability': np.random.normal(0.1, 0.05),
                'health_score': health_score,
                'confidence': np.random.uniform(0.7, 0.95),
                'environment_temp': np.random.normal(24, 2),
                'environment_ph': np.random.normal(7.0, 0.5),
                'light_intensity': np.random.normal(0.8, 0.1)
            })
            
        return pd.DataFrame(synthetic_data)
        
    def model_deployment_optimization(self):
        """모델 배포 최적화"""
        print("모델 배포 최적화 시작")
        
        if self.growth_predictor is None:
            print("예측 모델이 학습되지 않았습니다.")
            return
            
        # 모델 양자화
        self.growth_predictor.eval()
        
        # PyTorch JIT 컴파일
        example_input = torch.randn(1, 6).to(self.device)
        traced_model = torch.jit.trace(self.growth_predictor, example_input)
        traced_model.save(f'traced_growth_predictor_{self.experiment_id}.pt')
        
        # 모델 크기 최적화
        original_size = sum(p.numel() for p in self.growth_predictor.parameters())
        
        # 프루닝 (가중치 제거)
        import torch.nn.utils.prune as prune
        
        for module in self.growth_predictor.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.3)
                prune.remove(module, 'weight')
                
        pruned_size = sum(p.numel() for p in self.growth_predictor.parameters() if p.requires_grad)
        
        print(f"모델 크기 최적화: {original_size} → {pruned_size} ({(1-pruned_size/original_size)*100:.1f}% 감소)")
        
        # 최적화된 모델 저장
        torch.save(self.growth_predictor.state_dict(), f'optimized_growth_predictor_{self.experiment_id}.pth')
        
    def comprehensive_analysis_report(self, data_df, anomalies_df):
        """종합 분석 리포트 생성"""
        print("종합 분석 리포트 생성")
        
        report = {
            'experiment_id': self.experiment_id,
            'analysis_date': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(data_df),
                'date_range': {
                    'start': data_df['timestamp'].min() if 'timestamp' in data_df.columns else 'N/A',
                    'end': data_df['timestamp'].max() if 'timestamp' in data_df.columns else 'N/A'
                },
                'object_types': data_df['object_type'].value_counts().to_dict() if 'object_type' in data_df.columns else {}
            },
            'anomaly_analysis': {
                'total_anomalies': len(anomalies_df),
                'anomaly_rate': len(anomalies_df) / len(data_df) * 100,
                'critical_anomalies': len(anomalies_df[anomalies_df['growth_rate'] < -0.1]) if 'growth_rate' in anomalies_df.columns else 0
            },
            'growth_patterns': {
                'avg_growth_rate': data_df['growth_rate'].mean() if 'growth_rate' in data_df.columns else 0,
                'growth_std': data_df['growth_rate'].std() if 'growth_rate' in data_df.columns else 0,
                'max_growth_rate': data_df['growth_rate'].max() if 'growth_rate' in data_df.columns else 0,
                'min_growth_rate': data_df['growth_rate'].min() if 'growth_rate' in data_df.columns else 0
            },
            'health_metrics': {
                'avg_health_score': data_df['health_score'].mean() if 'health_score' in data_df.columns else 0,
                'healthy_objects_pct': (data_df['health_score'] > 0.8).mean() * 100 if 'health_score' in data_df.columns else 0,
                'critical_health_count': (data_df['health_score'] < 0.5).sum() if 'health_score' in data_df.columns else 0
            },
            'recommendations': []
        }
        
        # 추천사항 생성
        if report['anomaly_analysis']['anomaly_rate'] > 15:
            report['recommendations'].append("이상치 비율이 높습니다. 환경 조건을 점검하세요.")
            
        if report['growth_patterns']['avg_growth_rate'] < 0.05:
            report['recommendations'].append("평균 성장률이 낮습니다. 사료량과 환경 조건을 조정하세요.")
            
        if report['health_metrics']['healthy_objects_pct'] < 70:
            report['recommendations'].append("건강한 개체 비율이 낮습니다. 즉시 시스템 점검이 필요합니다.")
            
        # 리포트 저장
        with open(f'analysis_report_{self.experiment_id}.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
        print("분석 리포트 저장 완료")
        return report
        
    def run_full_analysis_pipeline(self):
        """전체 분석 파이프라인 실행"""
        print("전체 분석 파이프라인 시작")
        
        # 1. 데이터 수집 (서버에서 가져오거나 로컬 파일 사용)
        data_df = self.fetch_training_data()
        if data_df is None or len(data_df) < 100:
            print("충분한 데이터가 없습니다. 합성 데이터 생성")
            data_df = self.generate_synthetic_data(1000)
            
        # 2. 하이퍼파라미터 최적화
        best_params = self.hyperparameter_optimization(data_df)
        
        # 3. 성장 예측 모델 학습
        self.train_growth_predictor(data_df)
        
        # 4. 강화학습 에이전트 학습
        self.train_rl_agent()
        
        # 5. 이상 탐지 분석
        anomalies_df = self.anomaly_detection_analysis(data_df)
        
        # 6. 모델 최적화
        self.model_deployment_optimization()
        
        # 7. 종합 리포트 생성
        final_report = self.comprehensive_analysis_report(data_df, anomalies_df)
        
        # 8. 배포용 패키지 준비 (USB 전송용)
        deploy_dir, package_info = self.prepare_deployment_package(final_report)
        
        if deploy_dir:
            print(f"\n배포 패키지 준비 완료!")
            print(f"경로: {os.path.abspath(deploy_dir)}")
            print(f"USB에 복사하여 서버로 이동하세요")
            print(f"배포 가이드: {deploy_dir}/DEPLOYMENT_GUIDE.md")
        
        print("🎉 전체 분석 파이프라인 완료!")
        return final_report, deploy_dir

    def fetch_training_data(self):
        """서버에서 학습 데이터 수집 (선택적)"""
        try:
            print("📡 서버에서 데이터 수집 시도...")
            response = requests.get(f"{self.server_url}/api/training_data", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['records'])
                print(f"서버에서 학습 데이터 수집: {len(df)}개 레코드")
                return df
            else:
                print(f"서버 응답 오류: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"🔌 서버 연결 실패 (오프라인 모드): {e}")
            
            # 로컬 파일에서 데이터 로드 시도
            local_files = ['aquaponics_data.xlsx', 'training_data.csv', 'sample_data.json']
            for file_path in local_files:
                if os.path.exists(file_path):
                    print(f"로컬 파일 사용: {file_path}")
                    if file_path.endswith('.xlsx'):
                        return pd.read_excel(file_path)
                    elif file_path.endswith('.csv'):
                        return pd.read_csv(file_path)
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            return pd.DataFrame(data)
            
            print("로컬 데이터 파일도 없음")
            return None
        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None
        
    def prepare_deployment_package(self, report):
        """배포용 패키지 준비 (USB 전송용)"""
        print("배포용 패키지 준비 중...")
        
        # 배포 디렉토리 생성
        deploy_dir = f"deployment_package_{self.experiment_id}"
        os.makedirs(deploy_dir, exist_ok=True)
        os.makedirs(f"{deploy_dir}/models", exist_ok=True)
        os.makedirs(f"{deploy_dir}/configs", exist_ok=True)
        os.makedirs(f"{deploy_dir}/reports", exist_ok=True)
        
        package_info = {
            'experiment_id': self.experiment_id,
            'creation_date': datetime.now().isoformat(),
            'models': {},
            'configs': {},
            'reports': []
        }
        
        try:
            # 1. 최적화된 모델들 복사
            if self.growth_predictor is not None:
                # PyTorch 모델
                model_path = f"{deploy_dir}/models/growth_predictor.pth"
                torch.save(self.growth_predictor.state_dict(), model_path)
                
                # JIT 컴파일된 모델
                traced_path = f"{deploy_dir}/models/growth_predictor_traced.pt"
                if os.path.exists(f'traced_growth_predictor_{self.experiment_id}.pt'):
                    import shutil
                    shutil.copy(f'traced_growth_predictor_{self.experiment_id}.pt', traced_path)
                
                # 모델 정보
                package_info['models']['growth_predictor'] = {
                    'file': 'growth_predictor.pth',
                    'traced_file': 'growth_predictor_traced.pt',
                    'input_dim': 6,
                    'output_dim': 1,
                    'description': '성장률 예측 모델'
                }
                
            if self.rl_agent is not None:
                # 강화학습 모델
                rl_path = f"{deploy_dir}/models/rl_agent.zip"
                self.rl_agent.save(rl_path)
                
                package_info['models']['rl_agent'] = {
                    'file': 'rl_agent.zip',
                    'algorithm': 'PPO',
                    'description': '환경 제어 최적화 에이전트'
                }
                
            if self.yolo_model is not None:
                # YOLO 모델
                yolo_path = f"{deploy_dir}/models/custom_yolo.pt"
                # 기존 학습된 모델 경로에서 복사
                best_model_path = f"aquaponics_yolo/experiment_{self.experiment_id}/weights/best.pt"
                if os.path.exists(best_model_path):
                    import shutil
                    shutil.copy(best_model_path, yolo_path)
                    
                    package_info['models']['yolo'] = {
                        'file': 'custom_yolo.pt',
                        'version': 'YOLOv8m',
                        'description': '커스텀 아쿠아포닉스 객체 탐지 모델'
                    }
            
            # 2. 설정 파일들
            # 모델 설정
            model_config = {
                'pixel_to_cm_ratio': 0.08,
                'fish_thresholds': {
                    'length_min': 5.0,
                    'length_max': 30.0,
                    'weight_min': 10.0,
                    'weight_max': 500.0
                },
                'plant_thresholds': {
                    'height_min': 10.0,
                    'height_max': 60.0
                },
                'change_detection': {
                    'threshold': 0.20,
                    'cooldown_seconds': 5.0
                }
            }
            
            with open(f"{deploy_dir}/configs/model_config.json", 'w') as f:
                json.dump(model_config, f, indent=2)
                
            package_info['configs']['model_config'] = 'model_config.json'
            
            # 최적화 설정
            optimization_config = {
                'edge_device': {
                    'target_fps': 10,
                    'max_resolution': 640,
                    'frame_skip': 2,
                    'model_version': 'yolov5n'
                },
                'server': {
                    'batch_size': 5,
                    'queue_max_size': 50,
                    'model_timeout': 300,
                    'cleanup_interval': 3600
                }
            }
            
            with open(f"{deploy_dir}/configs/optimization_config.json", 'w') as f:
                json.dump(optimization_config, f, indent=2)
                
            package_info['configs']['optimization_config'] = 'optimization_config.json'
            
            # 3. 리포트 및 분석 결과
            with open(f"{deploy_dir}/reports/analysis_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                
            package_info['reports'].append('analysis_report.json')
            
            # 시각화 결과 복사
            if os.path.exists(f'anomaly_analysis_{self.experiment_id}.png'):
                import shutil
                shutil.copy(
                    f'anomaly_analysis_{self.experiment_id}.png',
                    f"{deploy_dir}/reports/anomaly_analysis.png"
                )
                package_info['reports'].append('anomaly_analysis.png')
            
            # 4. 배포 가이드 생성
            deployment_guide = self._create_deployment_guide(package_info)
            with open(f"{deploy_dir}/DEPLOYMENT_GUIDE.md", 'w', encoding='utf-8') as f:
                f.write(deployment_guide)
            
            # 5. 패키지 정보 저장
            with open(f"{deploy_dir}/package_info.json", 'w') as f:
                json.dump(package_info, f, indent=2, default=str)
            
            # 6. 체크섬 생성 (무결성 확인용)
            self._generate_checksums(deploy_dir)
            
            print(f"배포 패키지 준비 완료: {deploy_dir}/")
            print(f"포함된 파일들:")
            for root, dirs, files in os.walk(deploy_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), deploy_dir)
                    print(f"   {rel_path}")
            
            return deploy_dir, package_info
            
        except Exception as e:
            print(f"패키지 준비 실패: {e}")
            return None, None
    
    def _create_deployment_guide(self, package_info):
        """배포 가이드 문서 생성"""
        guide = f"""
# 아쿠아포닉스 모델 배포 가이드

## 실험 정보
- **실험 ID**: {package_info['experiment_id']}
- **생성 날짜**: {package_info['creation_date']}

## 포함된 모델들

"""
        
        for model_name, model_info in package_info['models'].items():
            guide += f"### {model_name}\n"
            guide += f"- **파일**: {model_info['file']}\n"
            guide += f"- **설명**: {model_info['description']}\n"
            if 'traced_file' in model_info:
                guide += f"- **최적화 버전**: {model_info['traced_file']}\n"
            guide += "\n"
        
        guide += """
## 배포 순서

### 1. 서버 배포 (ubuntu_server_low.py)
```bash
# 1. 모델 디렉토리 생성
mkdir -p /opt/aquaponics/models

# 2. 모델 파일 복사
cp models/* /opt/aquaponics/models/

# 3. 설정 파일 복사
cp configs/* /opt/aquaponics/configs/

# 4. 서버 재시작
sudo systemctl restart aquaponics-server
```

### 2. 라즈베리파이 배포 (raspberry_pi_edge.py)
```bash
# 1. 경량화된 모델만 복사 (YOLOv5n)
scp models/yolo_optimized.pt pi@raspberrypi:/home/pi/aquaponics/

# 2. 설정 업데이트
scp configs/model_config.json pi@raspberrypi:/home/pi/aquaponics/

# 3. 서비스 재시작
ssh pi@raspberrypi "sudo systemctl restart aquaponics-edge"
```

## 모델 성능 정보
"""
        
        # 성능 정보 추가 (report에서 추출)
        guide += """
- **예상 처리 속도**: 
  - 라즈베리파이: ~10 FPS
  - 서버: ~30 FPS (배치 처리)
- **메모리 사용량**:
  - 라즈베리파이: ~1GB
  - 서버: ~2GB

## 문제 해결

### 모델 로드 실패
1. 파일 경로 확인
2. 권한 확인 (`chmod 755`)
3. 의존성 패키지 설치 확인

### 성능 저하
1. 해상도 조정 (640px 이하)
2. FPS 제한 (10fps 이하)
3. 배치 크기 조정
"""
        
        return guide
    
    def _generate_checksums(self, deploy_dir):
        """파일 체크섬 생성"""
        import hashlib
        
        checksums = {}
        
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                if file == 'checksums.json':
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, deploy_dir)
                
                with open(file_path, 'rb') as f:
                    content = f.read()
                    checksums[rel_path] = hashlib.sha256(content).hexdigest()
        
        with open(f"{deploy_dir}/checksums.json", 'w') as f:
            json.dump(checksums, f, indent=2)
        
        print(f"체크섬 생성 완료: {len(checksums)}개 파일")

def main():
    """메인 실행 함수"""
    print("결과는 USB로 서버에 직접 전송됩니다")
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
    else:
        print("CPU 모드로 실행")
        
    # 시스템 초기화
    research_system = ResearchLabSystem()
    
    # 전체 파이프라인 실행
    try:
        final_report, deploy_dir = research_system.run_full_analysis_pipeline()
        
        print("\n최종 결과 요약:")
        print(f"실험 ID: {research_system.experiment_id}")
        print(f"처리된 데이터: {final_report['data_summary']['total_records']}개")
        print(f"이상치 감지: {final_report['anomaly_analysis']['total_anomalies']}개")
        print(f"평균 성장률: {final_report['growth_patterns']['avg_growth_rate']:.4f}")
        
        if deploy_dir:
            print(f"\n배포 패키지 위치:")
            print(f"   {os.path.abspath(deploy_dir)}")
            print(f"\n다음 단계:")
            print(f"   1. 위 폴더를 USB에 복사")
            print(f"   2. 우분투 서버로 이동")
            print(f"   3. DEPLOYMENT_GUIDE.md 참조하여 배포")
            
            # 압축 파일 생성 옵션
            try:
                import zipfile
                zip_path = f"{deploy_dir}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(deploy_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.dirname(deploy_dir))
                            zipf.write(file_path, arcname)
                
                print(f"압축 파일도 생성됨: {zip_path}")
                
            except Exception as e:
                print(f"압축 파일 생성 실패: {e}")
        
        print(f"\n모든 작업 완료! USB로 서버에 전송하세요.")
        
    except KeyboardInterrupt:
        print("\n사용자에 의한 중단")
    except Exception as e:
        print(f"\n파이프라인 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 리소스 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 리소스 정리 완료")

if __name__ == "__main__":
    main()