
def estimate_filter_for_echocanceller(L, mu, filterfile, floorfile, n_sec, 
                      freq, nch, tgt_chs, deviceid=None):
    import sounddevice as sd
    import numpy as np
    from lib.processor import LMSblockFFT

    nlimit = freq * n_sec
    outdata = np.random.randn(nlimit,1).astype('float32')

    print('[LOG]: now recording ...')
    if deviceid is not None:
        sd.default.device = deviceid
    wavdata = sd.playrec(outdata, freq, channels=nch, blocking=True)

    print('[LOG]: now estimating filter ...')    
    # lazy implementation :P
    lms = LMSblockFFT(L, mu)
    errs = []
    for idx in range(0, len(wavdata), 1):
        errs.append(lms.update(wavdata[idx:idx+1,:], False))
    errs = np.concatenate(errs)
    
    lms.save(filterfile)
    np.savetxt(floorfile, np.array([np.var(errs[-freq*int(n_sec/2):])]))

    print(f'[LOG]: save to {filterfile} and {floorfile}')


def usage():
    """
    return
        args: argparse.Namespece
    """
    import argparse

    # argument analysis
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('mode', type=str)

    ### calibration of filter
    parser.add_argument('--filter_filename', type=str, default='conf/ecfilter.txt',
                        help='filename for the estimated filter of echo canceller')
    parser.add_argument('--floor_filename', type=str, default='conf/ecfloor.txt',
                        help='filename for the estimated variance of error signal')
    parser.add_argument('--L', type=int, default=512,
                        help='filter length of echo canceller')
    parser.add_argument('--mu', type=float, default=0.005,
                        help='learning rate of echo canceller')
    parser.add_argument('--deviceid', type=int, default=None,
                        help='device id')
    parser.add_argument('--freq', type=int, default=16000,
                        help='sampling frequency')
    parser.add_argument('--nch', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--nsec', type=int, default=10,
                        help='recording time')
    parser.add_argument('--tgt_chs', type=int, nargs="*", default=[0,1], 
                        help='channel list of audio source for processing')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = usage()

    if args.mode == 'devinfo':
        import lib.io
        devicelist = lib.io.SoundDeviceSource.query_device()
        print('[LOG]: devicelist')
        print(devicelist)
    elif args.mode == 'calib_filter':
        print('[LOG]:', args)
        print('[LOG]: calibrate filter')
        estimate_filter_for_echocanceller(args.L, args.mu, args.filter_filename, args.floor_filename,
                          args.nsec, args.freq, args.nch, args.tgt_chs, args.deviceid) 
    pass
