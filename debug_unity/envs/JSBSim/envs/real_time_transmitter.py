# real_time_transmitter.py
import socket
import threading
import queue
import time
import logging
from typing import Optional, List

class RealTimeACMITransmitter:
    """
    实时ACMI数据传输器
    用于在训练过程中实时向Unity传输ACMI格式的仿真数据
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5000):
        """
        初始化传输器
        
        Args:
            host (str): Unity监听的IP地址
            port (int): Unity监听的端口号
        """
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.client_connection: Optional[socket.socket] = None
        self.data_queue = queue.Queue()  # 线程安全的数据队列
        self.is_running = False
        self.transmit_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
    def start_server(self):
        """
        启动TCP服务器，等待Unity连接
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.logger.info(f"ACMI实时传输服务器已启动: {self.host}:{self.port}")
            print(f"等待Unity连接: {self.host}:{self.port}")
            
            # 等待Unity连接
            self.client_connection, addr = self.server_socket.accept()
            self.logger.info(f"Unity已连接: {addr}")
            print(f"Unity已连接: {addr}")
            
            # 启动数据传输线程
            self.is_running = True
            self.transmit_thread = threading.Thread(target=self._transmit_worker, daemon=True)
            self.transmit_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动服务器失败: {e}")
            print(f"启动服务器失败: {e}")
            return False
    
    def _transmit_worker(self):
        """
        数据传输工作线程
        从队列中取出数据并发送给Unity
        """
        while self.is_running and self.client_connection:
            try:
                # 从队列获取数据，设置超时避免阻塞
                data = self.data_queue.get(timeout=1.0)
                
                if data is None:  # 停止信号
                    break
                
                # 发送数据到Unity
                self._send_data(data)
                self.data_queue.task_done()
                
            except queue.Empty:
                continue  # 队列为空，继续等待
            except (ConnectionResetError, BrokenPipeError) as e:
                self.logger.warning(f"Unity连接断开: {e}")
                print("Unity连接断开")
                break
            except Exception as e:
                self.logger.error(f"数据传输错误: {e}")
                break
    
    def _send_data(self, data: str):
        """
        发送数据到Unity客户端
        
        Args:
            data (str): 要发送的ACMI格式数据
        """
        try:
            # 编码数据
            data_bytes = data.encode('utf-8')
            
            # 发送长度头（4字节大端序）+ 实际数据
            length_header = len(data_bytes).to_bytes(4, 'big')
            self.client_connection.sendall(length_header + data_bytes)
            
            self.logger.debug(f"发送数据: {len(data_bytes)} 字节")
            
        except Exception as e:
            self.logger.error(f"发送数据失败: {e}")
            raise
    
    def send_frame_data(self, frame_data: List[str]):
        """
        向传输队列添加一帧数据
        
        Args:
            frame_data (List[str]): 一帧的ACMI数据，包含时间戳和实体数据
        """
        if not self.is_running or not self.client_connection:
            return False
        
        try:
            # 格式化数据
            formatted_data = "\n".join(frame_data) + "\n"
            
            # 添加到队列（非阻塞）
            self.data_queue.put_nowait(formatted_data)
            return True
            
        except queue.Full:
            self.logger.warning("数据队列已满，丢弃数据")
            return False
        except Exception as e:
            self.logger.error(f"添加数据到队列失败: {e}")
            return False
    
    def stop(self):
        """
        停止传输服务
        """
        self.logger.info("正在停止ACMI传输服务...")
        
        self.is_running = False
        
        # 发送停止信号到工作线程
        try:
            self.data_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # 等待工作线程结束
        if self.transmit_thread and self.transmit_thread.is_alive():
            self.transmit_thread.join(timeout=2.0)
        
        # 关闭连接
        if self.client_connection:
            try:
                self.client_connection.close()
            except:
                pass
            self.client_connection = None
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        self.logger.info("ACMI传输服务已停止")
        print("ACMI传输服务已停止")
    
    def is_connected(self) -> bool:
        """
        检查是否与Unity保持连接
        
        Returns:
            bool: 连接状态
        """
        return self.is_running and self.client_connection is not None