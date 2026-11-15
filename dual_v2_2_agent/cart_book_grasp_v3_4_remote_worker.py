"""Camera-only SSH worker; head subscribes to an existing RGB-D owner."""
import hashlib
import ctypes
import ctypes.util
from functools import lru_cache
import io
import json
import os
import sys
import time
import urllib.request

MAX_BYTES = 64*1024*1024


@lru_cache(maxsize=1)
def zstd_library():
    path = ctypes.util.find_library('zstd')
    if path is None:
        raise ValueError('System libzstd is required for lossless camera transport')
    library = ctypes.CDLL(path)
    signatures = {
        'ZSTD_compressBound': ([ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_compress': ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_size_t),
        'ZSTD_decompress': ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_getFrameContentSize': ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ulonglong),
        'ZSTD_findFrameCompressedSize': ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_isError': ([ctypes.c_size_t], ctypes.c_uint),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return library


def compress_payload(payload):
    library = zstd_library()
    output = ctypes.create_string_buffer(library.ZSTD_compressBound(len(payload)))
    size = library.ZSTD_compress(output, len(output), payload, len(payload), 3)
    if library.ZSTD_isError(size):
        raise ValueError('Camera lossless compression failed')
    return output.raw[:size]


def decompress_payload(payload):
    library = zstd_library()
    size = library.ZSTD_getFrameContentSize(payload, len(payload))
    if not 0 < size <= MAX_BYTES or library.ZSTD_findFrameCompressedSize(payload, len(payload)) != len(payload):
        raise ValueError('Invalid or oversized compressed camera payload')
    output = ctypes.create_string_buffer(size)
    actual = library.ZSTD_decompress(output, size, payload, len(payload))
    if library.ZSTD_isError(actual) or actual != size:
        raise ValueError('Corrupt compressed camera payload')
    return output.raw


def reply(request, payload, age, received=None):
    if len(payload) > MAX_BYTES:
        raise ValueError('Camera payload too large')
    started = time.monotonic()
    payload = compress_payload(payload)
    digest = hashlib.sha256(payload).hexdigest()
    finished = time.monotonic()
    header = dict(nonce=request['nonce'], bytes=len(payload),
                  encoding='zstd', sha256=digest,
                  source_age_s=age+finished-started,
                  processing_s=finished-received if received is not None else 0.)
    sys.stdout.buffer.write(json.dumps(header).encode()+b'\n'+payload)
    sys.stdout.buffer.flush()


def main():
    import numpy as np
    line = b''
    while not line.endswith(b'\n'):
        chunk = os.read(sys.stdin.fileno(), 1)
        if not chunk or len(line) >= 8192:
            raise ValueError('Missing or oversized camera configuration')
        line += chunk
    config = json.loads(line)
    try:
        if config['kind'] in ('head', 'wrist'):
            zstd_library()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if config['kind'] == 'wrist':
                from urllib.parse import urlencode
                query = {key: config[key] for key in ('device', 'width', 'height')}
                url = 'http://127.0.0.1:'+str(config['head_port'])+'/wrist?'+urlencode(query)
                with opener.open(url, timeout=3) as response:
                    if len(response.read(MAX_BYTES+1)) > MAX_BYTES:
                        raise ValueError('Oversized wrist warmup response')
            sys.stdout.buffer.write(b'{"ready":true}\n')
            sys.stdout.buffer.flush()
            for line in sys.stdin.buffer:
                received = time.monotonic()
                request = json.loads(line)
                from urllib.parse import urlencode
                if config['kind'] == 'head':
                    route, query = '/frame', {'after': request.get('after', '')}
                else:
                    route = '/wrist'
                    query = {key: config[key] for key in ('device', 'width', 'height')}
                url = 'http://127.0.0.1:'+str(config['head_port'])+route+'?'+urlencode(query)
                try:
                    with opener.open(url, timeout=3) as response:
                        payload = response.read(MAX_BYTES+1)
                except OSError as error:
                    raise ValueError(f'155 shared {config["kind"]} service unavailable on loopback port '
                                     f'{config["head_port"]}: {error}; no USB or preview fallback') from error
                if len(payload) > MAX_BYTES:
                    raise ValueError('Head RGB-D payload too large')
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    header = json.loads(str(archive['header']))
                age = time.time()-float(header['captured_unix_s'])
                if not 0 <= age <= .5:
                    raise ValueError('Remote head source timestamp stale or invalid')
                if config['kind'] == 'head':
                    if header.get('protocol') != 'head-rgbd-v1' or not header['metadata'].get('native_is_maximum'):
                        raise ValueError('Head source must publish highest native RGB-D, not a resized preview')
                    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                        native_bgr, native_depth = archive['native_bgr'], archive['native_depth']
                    header['transport_baseline_from_native'] = True
                    output = io.BytesIO()
                    np.savez(output, native_bgr=native_bgr, native_depth=native_depth,
                             header=np.array(json.dumps(header, allow_nan=False)))
                    payload = output.getvalue()
                elif (header.get('protocol') != 'wrist-rgb-v1' or any(
                        header['metadata'].get(key) != config[key] for key in ('device', 'width', 'height'))):
                    raise ValueError('Shared wrist identity/calibrated dimensions mismatch')
                reply(request, payload, time.time()-float(header['captured_unix_s']), received)
        else:
            raise ValueError('Unknown camera source')
    except Exception as error:
        sys.stdout.buffer.write(json.dumps({'error': str(error)}).encode()+b'\n')
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
