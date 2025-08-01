# common/mqtt_client.py - MQTT 클라이언트 공통 모듈
import json
import logging
import paho.mqtt.client as mqtt
from typing import Dict, Callable, Any
from datetime import datetime

class MQTTClient:
    """MQTT 클라이언트 공통 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config['mqtt']
        self.client = mqtt.Client()
        self.subscriptions = {}
        self.connected = False
        
        # 콜백 설정
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # 인증 설정
        if self.config.get('username'):
            self.client.username_pw_set(
                self.config['username'], 
                self.config.get('password', '')
            )
        
        self.connect()
    
    def connect(self):
        """MQTT 브로커 연결"""
        try:
            self.client.connect(
                self.config['broker_host'],
                self.config['broker_port'],
                60
            )
            self.client.loop_start()
            logging.info(f"MQTT 연결 시도: {self.config['broker_host']}:{self.config['broker_port']}")
            
        except Exception as e:
            logging.error(f"MQTT 연결 오류: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """연결 콜백"""
        if rc == 0:
            self.connected = True
            logging.info("MQTT 연결 성공")
            
            # 기존 구독 복원
            for topic in self.subscriptions.keys():
                client.subscribe(f"{self.config['topics'][topic]}")
                
        else:
            self.connected = False 
            logging.error(f"MQTT 연결 실패: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """연결 해제 콜백"""
        self.connected = False
        logging.warning("MQTT 연결 해제")
    
    def _on_message(self, client, userdata, msg):
        """메시지 수신 콜백"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # 토픽별 핸들러 호출
            for sub_topic, handler in self.subscriptions.items():
                if self.config['topics'][sub_topic] in topic:
                    handler(topic, payload)
                    break
                    
        except Exception as e:
            logging.error(f"MQTT 메시지 처리 오류: {e}")
    
    def subscribe(self, topic_key: str, handler: Callable):
        """토픽 구독"""
        if topic_key in self.config['topics']:
            full_topic = self.config['topics'][topic_key]
            self.subscriptions[topic_key] = handler
            
            if self.connected:
                self.client.subscribe(full_topic)
                logging.info(f"MQTT 구독: {full_topic}")
        else:
            logging.error(f"알 수 없는 토픽: {topic_key}")
    
    def publish(self, topic_key: str, data: Dict) -> bool:
        """메시지 발행"""
        try:
            if topic_key in self.config['topics']:
                full_topic = self.config['topics'][topic_key]
                payload = json.dumps(data, default=str)
                
                result = self.client.publish(full_topic, payload, qos=1)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logging.debug(f"MQTT 발행 성공: {full_topic}")
                    return True
                else:
                    logging.error(f"MQTT 발행 실패: {result.rc}")
                    return False
            else:
                logging.error(f"알 수 없는 토픽: {topic_key}")
                return False
                
        except Exception as e:
            logging.error(f"MQTT 발행 오류: {e}")
            return False
    
    def disconnect(self):
        """연결 해제"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            logging.info("MQTT 연결 해제")
