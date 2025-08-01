# edge/camera_manager.py - 카메라 관리자
import cv2
import numpy as np
import logging
import threading
import time
from typing import Optional
import queue

class CameraManager:
    """카메라 관리자 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config['edge']['camera']
        self.rtsp_url = self.config['rtsp_url']
        self.resolution = tuple(self.config['resolution'])
        self.fps = self.config['fps']
        self.buffer_size = self.config['buffer_size']
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.running = False
        
        self.initialize_camera()
        
    def initialize_camera(self):
        """카메라 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                logging.error(f"카메라 연결 실패: {self.rtsp_url}")
                return
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # 백그라운드 캡처 시작
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logging.info(f"카메라 초기화 완료: {self.resolution}@{self.fps}fps")
            
        except Exception as e:
            logging.error(f"카메라 초기화 오류: {e}")
    
    def _capture_loop(self):
        """백그라운드 프레임 캡처"""
        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    # 큐가 가득 찬 경우 오래된 프레임 제거
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(frame)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"프레임 캡처 오류: {e}")
                time.sleep(1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """최신 프레임 획득"""
        try:
            frame = self.frame_queue.get(timeout=1)
            return frame
        except queue.Empty:
            return None
    
    def cleanup(self):
        """리소스 정리"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
            
        if self.cap:
            self.cap.release()
            
        logging.info("카메라 관리자 정리 완료")
