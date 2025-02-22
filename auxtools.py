

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


def main_calib_filter(args):
    print('[LOG]:', args)
    print('[LOG]: calibrate filter')
    estimate_filter_for_echocanceller(args.L, args.mu, args.filter_filename, args.floor_filename,
                                      args.nsec, args.freq, args.nch, args.tgt_chs, args.deviceid)


def estimate_framepower_for_threshold(n_sec, freq, default_pow_dur=20, n_fwd=0, n_bwd=50, deviceid=None, nch=1):
    import sounddevice as sd
    import numpy as np
    import torch
    from usr.fdvad import BufferedWav2AmpSpec

    nlimit = freq * n_sec
    pow_dur = default_pow_dur if n_bwd > default_pow_dur else n_bwd

    print('[LOG]: now recording ...')
    if deviceid is not None:
        sd.default.device = deviceid
    wavdata = sd.rec(nlimit, freq, channels=nch, blocking=True)
    wav2aspec = BufferedWav2AmpSpec(n_fwd, n_bwd)
    
    feats = wav2aspec.get(torch.tensor(wavdata).reshape(1,-1), n_sec)
    sum_feats = torch.mean(torch.sum(feats[:,-pow_dur:,:]**2, dim=2),dim=1)

    n_len = len(sum_feats)
    beg_frame = int(n_len * 0.2)
    end_frame = int(n_len * 0.8)

    print(f'[LOG]: estimated frame-power:',
          f'mean: {torch.mean(sum_feats[beg_frame:end_frame]).item():.4f},',
          f'std: {torch.std(sum_feats[beg_frame:end_frame]).item():.4f}')


def main_calib_framepower(args):
    print('[LOG]:', args)
    print('[LOG]: calibrate power')
    estimate_framepower_for_threshold(args.nsec, args.freq, args.default_pow_dur,
                                      args.n_fwd, args.n_bwd,
                                      deviceid=args.deviceid, nch=args.nch)


def main_devinfo(args):
    import lib.io
    devicelist = lib.io.SoundDeviceSource.query_device()
    print('[LOG]: devicelist')
    print(devicelist)


def usage():
    """
    return
        args: argparse.Namespece
    """
    import argparse

    # argument analysis
    parser = argparse.ArgumentParser()

    # required
    #parser.add_argument('mode', type=str)

    ###
    subparsers = parser.add_subparsers()

    ###
    parser_dev = subparsers.add_parser('devinfo')
    parser_dev.set_defaults(func=main_devinfo)
                           
    ### calibration of filter
    parser_ec = subparsers.add_parser('calib_filter')
    parser_ec.add_argument('--filter_filename', type=str, default='conf/ecfilter.txt',
                        help='filename for the estimated filter of echo canceller')
    parser_ec.add_argument('--floor_filename', type=str, default='conf/ecfloor.txt',
                        help='filename for the estimated variance of error signal')
    parser_ec.add_argument('--L', type=int, default=512,
                        help='filter length of echo canceller')
    parser_ec.add_argument('--mu', type=float, default=0.005,
                        help='learning rate of echo canceller')
    parser_ec.add_argument('--deviceid', type=int, default=None,
                        help='device id')
    parser_ec.add_argument('--freq', type=int, default=16000,
                        help='sampling frequency')
    parser_ec.add_argument('--nch', type=int, default=2,
                        help='number of channels')
    parser_ec.add_argument('--nsec', type=int, default=10,
                        help='recording time')
    parser_ec.add_argument('--tgt_chs', type=int, nargs="*", default=[0,1], 
                        help='channel list of audio source for processing')
    parser_ec.set_defaults(func=main_calib_filter)

    ### calibration of frame power
    parser_fp = subparsers.add_parser('calib_framepower')
    parser_fp.add_argument('--deviceid', type=int, default=None,
                        help='device id')
    parser_fp.add_argument('--default_pow_dur', type=int, default=20,
                        help='the number of frames for frame-power average calculation')
    parser_fp.add_argument('--nsec', type=int, default=10,
                        help='recording time')
    parser_fp.add_argument('--freq', type=int, default=16000,
                           help='sampling frequency')
    parser_fp.add_argument('--nch', type=int, default=1,
                        help='number of channels')
    parser_fp.add_argument('--n_fwd', type=int, default=0,
                        help='forward frames in FD-VAD based on DNN-HMM')
    parser_fp.add_argument('--n_bwd', type=int, default=50,
                        help='backward frames in FD-VAD based on DNN-HMM')
    parser_fp.set_defaults(func=main_calib_framepower)

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = usage()
    args.func(args)
