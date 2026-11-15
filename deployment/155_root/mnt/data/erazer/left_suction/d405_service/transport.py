"""Latest-frame transport copied from shared_rgbd_v34.Latest; no shared runtime imports."""
import io,json,threading,time,uuid
import numpy as np
MAX_AGE=.5
MAX_BYTES=64*1024*1024

class Latest:
    def __init__(self):
        self.condition = threading.Condition()
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.frame = None
        self.error = 'Waiting for camera'

    def publish(self, arrays, stamp, metadata):
        if not 0 <= time.time()-stamp <= MAX_AGE:
            raise ValueError('Camera frame is stale')
        with self.condition:
            self.sequence += 1
            self.frame = (f'{self.session}:{self.sequence}', stamp, arrays, metadata)
            self.error = ''
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = str(error)
            self.frame = None
            self.condition.notify_all()

    def get(self, after='', newer_than=0., timeout=1.):
        with self.condition:
            def ready():
                return (self.frame is not None and self.frame[0] != after and self.frame[1] >= newer_than
                        and 0 <= time.time()-self.frame[1] <= MAX_AGE)
            if not self.condition.wait_for(ready, timeout):
                raise TimeoutError(self.error or 'Waiting for fresh camera frame')
            token, stamp, arrays, metadata = self.frame
        header = dict(protocol='head-rgbd-v1' if 'depth' in arrays else 'wrist-rgb-v1',
                      token=token, captured_unix_s=stamp, metadata=metadata,
                      color_encoding='bgr8' if 'depth' in arrays else 'rgb8',
                      depth_encoding='32FC1', depth_units='mm')
        output = io.BytesIO()
        np.savez(output, header=np.array(json.dumps(header, allow_nan=False)), **arrays)
        if time.time()-stamp > MAX_AGE:
            raise TimeoutError('Camera frame expired during encoding')
        payload = output.getvalue()
        if len(payload) > MAX_BYTES:
            raise ValueError('Camera payload exceeds bounded transport')
        return payload

    def status(self):
        with self.condition:
            age = time.time()-self.frame[1] if self.frame else None
            return dict(sequence=self.sequence, frame_age_s=age, error=self.error,
                        live=age is not None and 0 <= age <= MAX_AGE,
                        metadata=self.frame[3] if self.frame else None)
