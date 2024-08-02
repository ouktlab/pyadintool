import lib.adinserver


def espnet_main(device):
    """
    device: str
        cpu or cuda
    """
    ############################
    import numpy as np
    from pathlib import Path
    from typing import Any, List, Optional, Sequence, Tuple, Union
    from espnet2.bin.asr_inference_streaming import Speech2TextStreaming

    '''
    Streaming Interface for implementing 'from_pretrained' method
    which has not been implemented in ESPnet (pip version).
    Note that the original code of 'from_pretrained' is implemented for Speech2Text class in ESPnet.
      line 670 @ https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_inference.py
      (ESPnet https://github.com/espnet/espnet, Apache License by Shinji Watanabe)
    We used it for Speech2TextStreaming class here.
    '''
    class Speech2TextStreamingInterface(Speech2TextStreaming):
        def __init__(self, **kwargs):
            super.__init__(kwargs)
        
        @staticmethod
        def from_pretrained(
                model_tag: Optional[str] = None,
                **kwargs: Optional[Any],
        ):
            if model_tag is not None:
                try:
                    from espnet_model_zoo.downloader import ModelDownloader
                except ImportError:
                    print(
                        "`espnet_model_zoo` is not installed. "
                        "Please install via `pip install -U espnet_model_zoo`."
                    )
                    raise
                d = ModelDownloader()
                kwargs.update(**d.download_and_unpack(model_tag))
            return Speech2TextStreaming(**kwargs)

    def loop(model, adinserver, np_dtype=np.float16, scale=32767.0):
        while True:
            try:
                isEOS, audio = adinserver.get()
        
                if isEOS is True:
                    print(f'[LOG]: end of audio segment')
                    results = model(
                        speech=np.empty(0, dtype=np_dtype),
                        is_final=True)
                    print(f'Result -- {results[0][0]}')
                    print(f'  score: {results[0][3][1].item():.2f},'
                          f' details: {results[0][3][2]}')
                else:
                    audio = audio.astype(np_dtype)/scale
                    results = model(speech=audio, is_final=False)
            except:
                break

        adinserver.stop()
        print("[LOG]: end")

    #############
    hfrepo = 'ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn'
    
    model = Speech2TextStreamingInterface.from_pretrained(
        hfrepo,
        device=device,
        token_type='char',
        bpemodel=None,
        maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=10,
        nbest=5,
        ctc_weight=0.3,
        lm_weight=0.1,
        penalty=0.0
    )

    hostname = 'localhost'
    port = 5530
    
    adinserver = lib.adinserver.AdinnetServer(hostname, port)
    adinserver.start()

    loop(model, adinserver)
    adinserver.join()


def whisper_main(device):
    """
    device: str
        cpu or cuda
    """
    ##########################
    import numpy as np
    from faster_whisper import WhisperModel

    def loop(model, adinserver):
        audio_segment = []
        while True:
            try:
                isEOS, audio = adinserver.get()
                if isEOS is None and audio is None:
                    break
        
                if isEOS is True:
                    print(f'[LOG]: end of audio segment')
                    segments, info = model.transcribe(
                        np.concatenate(audio_segment, axis=0),
                        beam_size=5, language="ja")
                    for segment in segments:
                        print("[%.2fs -> %.2fs] %s"
                              % (segment.start, segment.end, segment.text))
                    audio_segment = []
                else:
                    audio_segment.append(audio.astype(np.float16)/32767.0)
            except:
                break

        adinserver.stop()
        print("[LOG]: end")

    ############################
    model_size = "small"
    model = WhisperModel(model_size, device=device,
                         cpu_threads=8, compute_type="int8")
        
    hostname = 'localhost'
    port = 5530
    
    adinserver = lib.adinserver.AdinnetServer(hostname, port)
    adinserver.start()

    loop(model, adinserver)
    adinserver.join()


def usage():
    """
    return
        args: argparse.Namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='"ESPnet" or "Whisper"')
    parser.add_argument('--device', type=str, help='cpu, cuda', default='cpu')

    args = parser.parse_args()
    return args


def main():
    args = usage()

    if args.model == 'ESPnet':
        espnet_main(args.device)
    elif args.model == 'Whisper':
        whisper_main(args.device)
    pass


if __name__ == '__main__':
    main()
