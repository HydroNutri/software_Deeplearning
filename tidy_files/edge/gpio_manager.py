# edge/gpio_manager.py - GPIO 제어 관리자
import logging
import time
import threading
from typing import Dict, Optional, Union, List
from enum import Enum

logger = logging.getLogger(__name__)

class PinMode(Enum):
    """GPIO 핀 모드"""
    OUTPUT = "output"
    INPUT = "input"
    PWM = "pwm"
    SERVO = "servo"

class PinState(Enum):
    """GPIO 핀 상태"""
    LOW = 0
    HIGH = 1

class GPIOManager:
    """GPIO 제어 관리자 - 하드웨어 추상화"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pins = {}
        self.pwm_instances = {}
        self.servo_instances = {}
        self.gpio_available = False
        
        # GPIO 라이브러리 초기화
        self.init_gpio()
        
        logger.info(f"GPIO 관리자 초기화 완료 (사용 가능: {self.gpio_available})")
    
    def init_gpio(self):
        """GPIO 라이브러리 초기화"""
        try:
            # Raspberry Pi GPIO 시도
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.gpio_lib = "RPi.GPIO"
                
                # BCM 모드 설정
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                self.gpio_available = True
                logger.info("RPi.GPIO 라이브러리 로드 성공")
                
            except ImportError:
                # gpiozero 시도 (대안)
                try:
                    import gpiozero
                    self.gpiozero = gpiozero
                    self.gpio_lib = "gpiozero"
                    self.gpio_available = True
                    logger.info("gpiozero 라이브러리 로드 성공")
                    
                except ImportError:
                    logger.warning("GPIO 라이브러리를 찾을 수 없음 - 시뮬레이션 모드")
                    self.gpio_lib = "simulation"
                    
        except Exception as e:
            logger.error(f"GPIO 초기화 실패: {e}")
            self.gpio_lib = "simulation"
    
    def setup_pin(self, pin: int, mode: PinMode, **kwargs) -> bool:
        """GPIO 핀 설정"""
        try:
            if not self.gpio_available:
                logger.info(f"시뮬레이션: 핀 {pin} 설정 ({mode.value})")
                self.pins[pin] = {'mode': mode, 'state': PinState.LOW}
                return True
            
            if self.gpio_lib == "RPi.GPIO":
                if mode == PinMode.OUTPUT:
                    self.GPIO.setup(pin, self.GPIO.OUT)
                elif mode == PinMode.INPUT:
                    pull_up_down = kwargs.get('pull_up_down', self.GPIO.PUD_OFF)
                    self.GPIO.setup(pin, self.GPIO.IN, pull_up_down=pull_up_down)
                elif mode == PinMode.PWM:
                    frequency = kwargs.get('frequency', 1000)
                    self.GPIO.setup(pin, self.GPIO.OUT)
                    self.pwm_instances[pin] = self.GPIO.PWM(pin, frequency)
                elif mode == PinMode.SERVO:
                    self.GPIO.setup(pin, self.GPIO.OUT)
                    self.pwm_instances[pin] = self.GPIO.PWM(pin, 50)  # 50Hz for servo
                    
            elif self.gpio_lib == "gpiozero":
                if mode == PinMode.OUTPUT:
                    self.pins[pin] = self.gpiozero.LED(pin)
                elif mode == PinMode.INPUT:
                    self.pins[pin] = self.gpiozero.Button(pin)
                elif mode == PinMode.PWM:
                    self.pins[pin] = self.gpiozero.PWMLED(pin)
                elif mode == PinMode.SERVO:
                    self.pins[pin] = self.gpiozero.Servo(pin)
            
            self.pins[pin] = {'mode': mode, 'state': PinState.LOW}
            logger.debug(f"핀 {pin} 설정 완료: {mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"핀 {pin} 설정 실패: {e}")
            return False
    
    def digital_write(self, pin: int, state: Union[PinState, bool, int]) -> bool:
        """디지털 출력"""
        try:
            if isinstance(state, bool):
                state = PinState.HIGH if state else PinState.LOW
            elif isinstance(state, int):
                state = PinState.HIGH if state > 0 else PinState.LOW
            
            if not self.gpio_available:
                logger.debug(f"시뮬레이션: 핀 {pin} = {state.name}")
                if pin in self.pins:
                    self.pins[pin]['state'] = state
                return True
            
            if self.gpio_lib == "RPi.GPIO":
                gpio_state = self.GPIO.HIGH if state == PinState.HIGH else self.GPIO.LOW
                self.GPIO.output(pin, gpio_state)
                
            elif self.gpio_lib == "gpiozero":
                if pin in self.pins:
                    if state == PinState.HIGH:
                        self.pins[pin].on()
                    else:
                        self.pins[pin].off()
            
            if pin in self.pins:
                self.pins[pin]['state'] = state
                
            logger.debug(f"핀 {pin} 출력: {state.name}")
            return True
            
        except Exception as e:
            logger.error(f"핀 {pin} 디지털 출력 실패: {e}")
            return False
    
    def digital_read(self, pin: int) -> Optional[PinState]:
        """디지털 입력"""
        try:
            if not self.gpio_available:
                # 시뮬레이션: 랜덤 값 반환
                import random
                return PinState.HIGH if random.random() > 0.5 else PinState.LOW
            
            if self.gpio_lib == "RPi.GPIO":
                value = self.GPIO.input(pin)
                return PinState.HIGH if value else PinState.LOW
                
            elif self.gpio_lib == "gpiozero":
                if pin in self.pins:
                    return PinState.HIGH if self.pins[pin].is_pressed else PinState.LOW
            
            return PinState.LOW
            
        except Exception as e:
            logger.error(f"핀 {pin} 디지털 입력 실패: {e}")
            return None
    
    def pwm_write(self, pin: int, duty_cycle: float) -> bool:
        """PWM 출력 (0.0 ~ 1.0)"""
        try:
            duty_cycle = max(0.0, min(1.0, duty_cycle))  # 범위 제한
            
            if not self.gpio_available:
                logger.debug(f"시뮬레이션: 핀 {pin} PWM = {duty_cycle:.2f}")
                return True
            
            if self.gpio_lib == "RPi.GPIO":
                if pin in self.pwm_instances:
                    pwm = self.pwm_instances[pin]
                    if not hasattr(pwm, '_started'):
                        pwm.start(0)
                        pwm._started = True
                    pwm.ChangeDutyCycle(duty_cycle * 100)
                else:
                    logger.error(f"핀 {pin}이 PWM으로 설정되지 않음")
                    return False
                    
            elif self.gpio_lib == "gpiozero":
                if pin in self.pins:
                    self.pins[pin].value = duty_cycle
                else:
                    logger.error(f"핀 {pin}이 PWM으로 설정되지 않음")
                    return False
            
            logger.debug(f"핀 {pin} PWM 출력: {duty_cycle:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"핀 {pin} PWM 출력 실패: {e}")
            return False
    
    def servo_write(self, pin: int, angle: float) -> bool:
        """서보 모터 제어 (0 ~ 180도)"""
        try:
            angle = max(0, min(180, angle))  # 각도 제한
            
            if not self.gpio_available:
                logger.debug(f"시뮬레이션: 핀 {pin} 서보 = {angle}도")
                return True
            
            if self.gpio_lib == "RPi.GPIO":
                if pin in self.pwm_instances:
                    pwm = self.pwm_instances[pin]
                    if not hasattr(pwm, '_started'):
                        pwm.start(0)
                        pwm._started = True
                    
                    # 각도를 듀티 사이클로 변환 (1ms~2ms pulse width)
                    duty_cycle = 2.5 + (angle / 180.0) * 10.0
                    pwm.ChangeDutyCycle(duty_cycle)
                else:
                    logger.error(f"핀 {pin}이 서보로 설정되지 않음")
                    return False
                    
            elif self.gpio_lib == "gpiozero":
                if pin in self.pins:
                    # gpiozero 서보는 -1 ~ 1 범위
                    servo_value = (angle / 90.0) - 1.0
                    self.pins[pin].value = servo_value
                else:
                    logger.error(f"핀 {pin}이 서보로 설정되지 않음")
                    return False
            
            logger.debug(f"핀 {pin} 서보 제어: {angle}도")
            return True
            
        except Exception as e:
            logger.error(f"핀 {pin} 서보 제어 실패: {e}")
            return False
    
    def analog_read(self, pin: int, adc_type: str = "MCP3008") -> Optional[float]:
        """아날로그 입력 (ADC 필요)"""
        try:
            if not self.gpio_available:
                # 시뮬레이션: 랜덤 값 반환
                import random
                return random.uniform(0.0, 1.0)
            
            # MCP3008 ADC 사용 예시
            if adc_type == "MCP3008":
                try:
                    import spidev
                    
                    if not hasattr(self, 'spi'):
                        self.spi = spidev.SpiDev()
                        self.spi.open(0, 0)  # SPI bus 0, device 0
                        self.spi.max_speed_hz = 1350000
                    
                    # MCP3008에서 채널 읽기
                    adc_channel = pin % 8  # 0-7 채널
                    r = self.spi.xfer2([1, (8 + adc_channel) << 4, 0])
                    adc_value = ((r[1] & 3) << 8) + r[2]
                    
                    # 0-1023 값을 0.0-1.0으로 변환
                    return adc_value / 1023.0
                    
                except ImportError:
                    logger.warning("spidev 라이브러리 없음 - ADC 시뮬레이션")
                    import random
                    return random.uniform(0.0, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"핀 {pin} 아날로그 입력 실패: {e}")
            return None
    
    def cleanup_pin(self, pin: int):
        """개별 핀 정리"""
        try:
            if self.gpio_lib == "RPi.GPIO":
                if pin in self.pwm_instances:
                    self.pwm_instances[pin].stop()
                    del self.pwm_instances[pin]
                    
            if pin in self.pins:
                del self.pins[pin]
                
            logger.debug(f"핀 {pin} 정리 완료")
            
        except Exception as e:
            logger.error(f"핀 {pin} 정리 실패: {e}")
    
    def cleanup_all(self):
        """모든 GPIO 정리"""
        try:
            if self.gpio_lib == "RPi.GPIO":
                # 모든 PWM 정리
                for pwm in self.pwm_instances.values():
                    try:
                        pwm.stop()
                    except:
                        pass
                
                self.GPIO.cleanup()
                
            elif self.gpio_lib == "gpiozero":
                # gpiozero 객체들 정리
                for gpio_obj in self.pins.values():
                    if hasattr(gpio_obj, 'close'):
                        try:
                            gpio_obj.close()
                        except:
                            pass
            
            self.pins.clear()
            self.pwm_instances.clear()
            self.servo_instances.clear()
            
            logger.info("모든 GPIO 정리 완료")
            
        except Exception as e:
            logger.error(f"GPIO 정리 실패: {e}")
    
    def get_pin_status(self) -> Dict:
        """모든 핀의 현재 상태 조회"""
        status = {
            'gpio_available': self.gpio_available,
            'gpio_library': self.gpio_lib,
            'pins': {}
        }
        
        for pin, info in self.pins.items():
            status['pins'][pin] = {
                'mode': info['mode'].value if isinstance(info['mode'], PinMode) else str(info['mode']),
                'state': info['state'].name if isinstance(info['state'], PinState) else str(info['state'])
            }
        
        return status
    
    def test_pin(self, pin: int, mode: PinMode) -> bool:
        """핀 테스트"""
        try:
            # 핀 설정
            if not self.setup_pin(pin, mode):
                return False
            
            if mode == PinMode.OUTPUT:
                # 디지털 출력 테스트
                self.digital_write(pin, PinState.HIGH)
                time.sleep(0.1)
                self.digital_write(pin, PinState.LOW)
                
            elif mode == PinMode.PWM:
                # PWM 테스트
                for duty in [0.0, 0.5, 1.0, 0.0]:
                    self.pwm_write(pin, duty)
                    time.sleep(0.2)
                    
            elif mode == PinMode.SERVO:
                # 서보 테스트
                for angle in [0, 90, 180, 90]:
                    self.servo_write(pin, angle)
                    time.sleep(0.5)
                    
            elif mode == PinMode.INPUT:
                # 디지털 입력 테스트
                value = self.digital_read(pin)
                logger.info(f"핀 {pin} 입력값: {value}")
            
            logger.info(f"핀 {pin} 테스트 완료 ({mode.value})")
            return True
            
        except Exception as e:
            logger.error(f"핀 {pin} 테스트 실패: {e}")
            return False

# 편의 함수들
def create_gpio_manager(config: Dict) -> GPIOManager:
    """GPIO 관리자 생성"""
    return GPIOManager(config)

def test_all_pins(gpio_manager: GPIOManager, pin_configs: Dict) -> Dict:
    """모든 핀 테스트"""
    results = {}
    
    for pin, config in pin_configs.items():
        pin_num = int(pin)
        mode_str = config.get('type', 'output')
        
        # 문자열을 PinMode로 변환
        mode_mapping = {
            'relay': PinMode.OUTPUT,
            'pwm': PinMode.PWM,
            'servo': PinMode.SERVO,
            'input': PinMode.INPUT,
            'output': PinMode.OUTPUT
        }
        
        mode = mode_mapping.get(mode_str, PinMode.OUTPUT)
        results[pin_num] = gpio_manager.test_pin(pin_num, mode)
    
    return results
