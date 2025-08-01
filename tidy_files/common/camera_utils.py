# common/camera_utils.py - 카메라 관련 유틸리티 함수들
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_cameras() -> List[dict]:
    """사용 가능한 카메라 장치 감지"""
    cameras = []
    
    # USB 카메라 감지 (0-9번까지 확인)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                cameras.append({
                    'type': 'usb',
                    'index': i,
                    'device': f'/dev/video{i}',
                    'resolution': (width, height),
                    'status': 'available'
                })
            cap.release()
    
    # Raspberry Pi 카메라 감지
    try:
        import subprocess
        result = subprocess.run(['vcgencmd', 'get_camera'], 
                              capture_output=True, text=True)
        if 'detected=1' in result.stdout:
            cameras.append({
                'type': 'raspberry_pi',
                'index': 0,
                'device': '/dev/video0',
                'resolution': (1920, 1080),  # 기본 해상도
                'status': 'available'
            })
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # vcgencmd가 없거나 실행 실패
    
    logger.info(f"감지된 카메라: {len(cameras)}개")
    return cameras

def test_camera_connection(camera_source: str, timeout: int = 5) -> bool:
    """카메라 연결 테스트"""
    try:
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            return False
        
        # 타임아웃 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
        
    except Exception as e:
        logger.error(f"카메라 연결 테스트 실패: {e}")
        return False

def optimize_camera_settings(cap: cv2.VideoCapture, 
                           resolution: Tuple[int, int] = (640, 480),
                           fps: int = 10,
                           buffer_size: int = 1) -> bool:
    """카메라 설정 최적화"""
    try:
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # FPS 설정
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 버퍼 크기 최소화 (지연 감소)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        # 자동 노출 끄기 (일관된 조명)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        # 포커스 설정 (가능한 경우)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        logger.info(f"카메라 설정 완료: {resolution}@{fps}fps")
        return True
        
    except Exception as e:
        logger.error(f"카메라 설정 실패: {e}")
        return False

def preprocess_frame(frame: np.ndarray, 
                    target_size: Tuple[int, int] = (640, 480),
                    normalize: bool = True,
                    enhance: bool = True) -> np.ndarray:
    """프레임 전처리"""
    try:
        if frame is None:
            return None
        
        # 크기 조정
        if frame.shape[:2] != target_size[::-1]:  # OpenCV는 (height, width)
            frame = cv2.resize(frame, target_size)
        
        # 이미지 향상
        if enhance:
            # 히스토그램 평활화 (조명 보정)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 노이즈 제거
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 정규화
        if normalize:
            frame = frame.astype(np.float32) / 255.0
            frame = (frame * 255).astype(np.uint8)
        
        return frame
        
    except Exception as e:
        logger.error(f"프레임 전처리 실패: {e}")
        return frame

def extract_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                padding: int = 10) -> Optional[np.ndarray]:
    """바운딩 박스에서 ROI 추출"""
    try:
        if frame is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 패딩 추가
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # ROI 추출
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        return roi
        
    except Exception as e:
        logger.error(f"ROI 추출 실패: {e}")
        return None

def save_frame(frame: np.ndarray, 
               save_path: str,
               quality: int = 90,
               add_timestamp: bool = True) -> bool:
    """프레임 저장"""
    try:
        if frame is None:
            return False
        
        # 저장 경로 확인
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프 추가
        if add_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path_parts = Path(save_path)
            save_path = str(path_parts.parent / f"{path_parts.stem}_{timestamp}{path_parts.suffix}")
        
        # 이미지 저장
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success = cv2.imwrite(save_path, frame, encode_params)
        
        if success:
            logger.debug(f"프레임 저장 완료: {save_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"프레임 저장 실패: {e}")
        return False

def create_comparison_image(original: np.ndarray, 
                          processed: np.ndarray,
                          detections: List = None) -> np.ndarray:
    """원본과 처리된 이미지 비교 화면 생성"""
    try:
        if original is None or processed is None:
            return None
        
        # 크기 맞추기
        h, w = original.shape[:2]
        processed_resized = cv2.resize(processed, (w, h))
        
        # 좌우로 결합
        comparison = np.hstack([original, processed_resized])
        
        # 탐지 결과 그리기 (있는 경우)
        if detections:
            for detection in detections:
                if hasattr(detection, 'bbox'):
                    x1, y1, x2, y2 = detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2
                    
                    # 원본 이미지에 그리기
                    cv2.rectangle(comparison, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 처리된 이미지에 그리기 (오프셋 추가)
                    cv2.rectangle(comparison, (x1 + w, y1), (x2 + w, y2), (0, 0, 255), 2)
                    
                    # 클래스 라벨
                    if hasattr(detection, 'class_name'):
                        cv2.putText(comparison, detection.class_name, 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 1)
        
        # 구분선 그리기
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        
        # 라벨 추가
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison
        
    except Exception as e:
        logger.error(f"비교 이미지 생성 실패: {e}")
        return original

def calculate_frame_metrics(frame: np.ndarray) -> dict:
    """프레임 품질 메트릭 계산"""
    try:
        if frame is None:
            return {}
        
        # 그레이스케일 변환
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 기본 통계
        height, width = gray.shape
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # 대비 측정 (Michelson contrast)
        contrast = (np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-8)
        
        # 선명도 측정 (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # 노이즈 측정 (추정)
        noise_estimate = np.std(laplacian)
        
        metrics = {
            'width': width,
            'height': height,
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'noise_estimate': float(noise_estimate),
            'quality_score': float(min(1.0, (contrast * sharpness) / (noise_estimate + 1)))
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"프레임 메트릭 계산 실패: {e}")
        return {}

def draw_detection_overlay(frame: np.ndarray, 
                         detections: List,
                         show_confidence: bool = True,
                         show_id: bool = True) -> np.ndarray:
    """탐지 결과를 프레임에 오버레이"""
    try:
        if frame is None or not detections:
            return frame
        
        overlay_frame = frame.copy()
        
        # 클래스별 색상
        colors = {
            'fish': (0, 255, 0),      # 초록색
            'plant': (0, 255, 255),   # 노란색
            'food': (255, 0, 0)       # 파란색
        }
        
        for detection in detections:
            if hasattr(detection, 'bbox') and hasattr(detection, 'class_name'):
                x1, y1, x2, y2 = detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2
                class_name = detection.class_name
                confidence = getattr(detection, 'confidence', 0.0)
                object_id = getattr(detection, 'object_id', '')
                
                # 바운딩 박스 색상
                color = colors.get(class_name, (255, 255, 255))
                
                # 바운딩 박스 그리기
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 텍스트 생성
                label_parts = [class_name]
                
                if show_confidence:
                    label_parts.append(f"{confidence:.2f}")
                
                if show_id and object_id:
                    label_parts.append(f"ID:{object_id}")
                
                label = " | ".join(label_parts)
                
                # 텍스트 배경
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(overlay_frame, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), 
                            color, -1)
                
                # 텍스트 그리기
                cv2.putText(overlay_frame, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 0, 0), 1)
                
                # 추가 정보 (건강도, 크기 등)
                info_lines = []
                
                if hasattr(detection, 'health_score') and detection.health_score:
                    info_lines.append(f"Health: {detection.health_score:.2f}")
                
                if hasattr(detection, 'length_cm') and detection.length_cm:
                    info_lines.append(f"Length: {detection.length_cm:.1f}cm")
                
                if hasattr(detection, 'height_cm') and detection.height_cm:
                    info_lines.append(f"Height: {detection.height_cm:.1f}cm")
                
                # 추가 정보 표시
                for i, info in enumerate(info_lines):
                    cv2.putText(overlay_frame, info,
                              (x1, y2 + 15 + i * 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                              color, 1)
        
        return overlay_frame
        
    except Exception as e:
        logger.error(f"탐지 오버레이 그리기 실패: {e}")
        return frame

def create_grid_view(frames: List[np.ndarray], 
                    labels: List[str] = None,
                    grid_size: Tuple[int, int] = None) -> np.ndarray:
    """여러 프레임을 격자로 배열"""
    try:
        if not frames or len(frames) == 0:
            return None
        
        # 유효한 프레임만 필터링
        valid_frames = [f for f in frames if f is not None]
        if not valid_frames:
            return None
        
        num_frames = len(valid_frames)
        
        # 격자 크기 자동 계산
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_frames)))
            rows = int(np.ceil(num_frames / cols))
        else:
            rows, cols = grid_size
        
        # 첫 번째 프레임 크기 기준으로 모든 프레임 리사이즈
        h, w = valid_frames[0].shape[:2]
        target_h, target_w = h // 2, w // 2  # 격자에서는 작게 표시
        
        resized_frames = []
        for frame in valid_frames:
            resized = cv2.resize(frame, (target_w, target_h))
            resized_frames.append(resized)
        
        # 빈 프레임으로 채우기
        while len(resized_frames) < rows * cols:
            empty_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            resized_frames.append(empty_frame)
        
        # 격자 생성
        grid_rows = []
        for i in range(rows):
            row_frames = resized_frames[i * cols:(i + 1) * cols]
            
            # 라벨 추가 (있는 경우)
            if labels and i * cols < len(labels):
                for j, frame in enumerate(row_frames):
                    label_idx = i * cols + j
                    if label_idx < len(labels):
                        cv2.putText(frame, labels[label_idx],
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2)
            
            row_image = np.hstack(row_frames)
            grid_rows.append(row_image)
        
        grid_image = np.vstack(grid_rows)
        
        return grid_image
        
    except Exception as e:
        logger.error(f"격자 뷰 생성 실패: {e}")
        return None

def validate_frame(frame: np.ndarray, 
                  min_size: Tuple[int, int] = (64, 64),
                  max_size: Tuple[int, int] = (4096, 4096)) -> bool:
    """프레임 유효성 검사"""
    try:
        if frame is None:
            return False
        
        if len(frame.shape) not in [2, 3]:
            return False
        
        h, w = frame.shape[:2]
        
        # 크기 체크
        if w < min_size[0] or h < min_size[1]:
            return False
        
        if w > max_size[0] or h > max_size[1]:
            return False
        
        # 빈 프레임 체크
        if np.all(frame == 0) or np.all(frame == 255):
            return False
        
        return True
        
    except Exception:
        return False
