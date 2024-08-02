import threading
import queue
import socket
import struct
import numpy as np
import select


class AdinnetServer(threading.Thread):
    """
    """
    def __init__(self, hostname, port, bin_dtype='int16'):
        """
        hostname: str
        port: int
        bin_dtype: str
        """
        super(AdinnetServer, self).__init__()
        self.is_finish = False
        self.sock = None
        self.client = None
        self.q = queue.Queue()  # tuple of (isEnd, audio) 

        self.hostname = hostname
        self.port = port
        self.bin_dtype = bin_dtype

    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.hostname, self.port))
        self.sock.listen(1)
        print(f"[LOG]: now waiting for connection from adinnet client")

    def stop(self):
        self.is_finish = True

    def close(self, signal=None, frame=None):
        self.q.put((None, None))
        if self.client is not None:
            self.client.close()
        if self.sock is not None:
            self.sock.close()

    def check_select(self, d):
        while True:
            r,_,_ = select.select([d], [], [], 1.0)
            if len(r) > 0:
                return True
            if self.is_finish is True:
                return False

    def receive(self):
        """
        receive data from client
        """
        while self.is_finish is False:
            if self.check_select(self.client) is False:
                return
                
            # receive byte size
            m_len = self.client.recv(4)            
            if len(m_len) == 0:
                print("[LOG]: connection shutdown")
                raise Exception("connection shutdown")
                break
            
            m_len = struct.unpack('<i', m_len)[0]

            # end of segment
            if m_len == 0:
                self.q.put((True, None))
                continue
            
            # receive byte sequence
            n_left = m_len
            while n_left > 0:
                if self.check_select(self.client) is False:
                    return
                
                bindata = self.client.recv(n_left)                
                n_left -= len(bindata)
                audio = np.frombuffer(bindata, dtype=self.bin_dtype)
                self.q.put((False, audio))
            
        pass

    def run(self):
        self.open()
        while self.is_finish is False:
            try:
                if self.check_select(self.sock) is False:
                    break
                self.client, self.remote_addr = self.sock.accept()
                print(f"[LOG]: accept connection from {self.remote_addr}")
                self.receive()        
            except:
                break        
        self.close()
        
    def get(self, timeout=None):
        return self.q.get(timeout=timeout)
