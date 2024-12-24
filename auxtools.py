"""
"""
def record_gaussnoise(L, n_sec, 
                      device_i, device_o, freq, nch, tgt_chs):
    import lib.io
    import numpy as np

    n_sec = 3
    n_frame = 512
    
    nlimit = freq * n_sec
    source = lib.io.SoundDeviceSource(device_i, freq, nch,
                                      tgt_chs=tgt_chs,
                                      nlimit=nlimit+freq*0.5)
    sink = lib.io.SoundDeviceSink(device_o, freq, 1, n_frame)

    ###
    print('[LOG]: start recording...')
    source.open()
    sink.open()

    outdata = np.random.rand(nlimit,1).astype('float32')
    sink.write(outdata)
    
    wavdata = []
    while (data := source.read()) is not None:
        wavdata.append(data)
        
    sink.close()
    source.close()

    print('[LOG]: finish recording')

    print(wavdata[0].shape)
    wavdata = np.concatenate(wavdata)
    print(wavdata.shape)

    import matplotlib.pyplot as plt
    plt.plot(wavdata)
    plt.show()
    
if __name__ == "__main__":
    L = 512
    device_i = 'default'
    device_o = 'default'
    freq = 16000
    nch = 1
    tgt_chs = [0]
    record_gaussnoise(L, device_i, device_o, freq, nch, tgt_chs)
    pass
