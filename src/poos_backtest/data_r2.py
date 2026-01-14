from __future__ import annotations
import logging
import os
from dataclasses import dataclass
import boto3

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class R2Client:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    prefix: str

    def enabled(self) -> bool:
        return all([self.endpoint, self.access_key_id, self.secret_access_key, self.bucket])

    def _client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
        )

    def upload_file(self, local_path: str, key: str) -> None:
        if not self.enabled():
            log.info("R2 not configured; skip upload: %s", local_path)
            return
        s3 = self._client()
        r2_key = f"{self.prefix.rstrip('/')}/{key.lstrip('/')}"
        log.info("Uploading to R2: s3://%s/%s", self.bucket, r2_key)
        s3.upload_file(local_path, self.bucket, r2_key)

def from_env() -> R2Client:
    return R2Client(
        endpoint=os.getenv("R2_ENDPOINT", ""),
        access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        bucket=os.getenv("R2_BUCKET", ""),
        prefix=os.getenv("R2_PREFIX", "poos-v1"),
    )
