import socket
import numpy as np
import struct

def adinnet(HOSTNAME, PORT):
    adinsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    adinsock.bind((HOSTNAME, PORT))
    adinsock.listen(1)

    print("[LOG]: now waiting for connection from mic.")
    adinclient, remote_addr = adinsock.accept()
    print(f"[LOG]: accept connection from {remote_addr}")

    n_total = 0
    while True:
        m_len = adinclient.recv(4)
        if len(m_len) == 0:
            print("[LOG]: connection from mic. is shutdown. exit.")
            break
        m_len = struct.unpack('<i', m_len)[0]

        # finish
        if m_len == 0:
            print(f'\n[LOG]: end of audio segment. received {n_total} samples.')
            n_total = 0
            pass        
        else:
            n_left = m_len
            packets = []
            while n_left > 0:
                data = adinclient.recv(n_left)
                n_left = n_left - len(data)
                wav = np.frombuffer(data, dtype='int16')
                packets.append(wav)
                
            wav = np.concatenate(packets, axis=0)
            print(f'\r[LOG]: received bytes: {m_len}', end='')
            n_total += wav.shape[0]
    

if __name__ == '__main__':
    adinnet('localhost', 5530)
    pass
