"""Local credentials and auditable, resumable live experiments (never logs headers)."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from .backend import JevBackend


def load_local_key(path: Path) -> None:
    """Explicit CLI opt-in; an existing environment variable always wins."""
    if os.environ.get('TYPESAFE_API_KEY') or not path.is_file():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        name, sep, value = line.strip().partition('=')
        if sep and name == 'TYPESAFE_API_KEY':
            os.environ[name] = value.strip().strip('\"\'')
            return


def digest(value) -> str:
    # Preserve option/candidate order: permutations must not become cache hits.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


class AuditedJevBackend(JevBackend):
    """Save exact public requests and complete responses; reuse only exact requests.

    The cache directory belongs to one frozen run. Separate directories are
    required for independent identical-call replications.
    """
    def __init__(self, *, cache_dir: Path, max_requests: int = 500,
                 max_input_tokens: int = 1000000, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_requests = max_requests
        self.max_input_tokens = max_input_tokens
        self.cache_hits = 0
        self.records = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_response = None
        self.resolved_models = set()
        # Persisted totals apply on resume, including earlier failed HTTP attempts.
        self.previous_requests = 0
        for file in cache_dir.glob('*.json'):
            record = json.loads(file.read_text(encoding='utf-8'))
            if record.get('response', {}).get('model'):
                self.resolved_models.add(record['response']['model'])
            self.previous_requests += record.get('http_attempts', 1)
            usage = record.get('response', {}).get('usage', {})
            self.input_tokens += usage.get('input_tokens', 0)
            self.output_tokens += usage.get('output_tokens', 0)
        self.failure_log = cache_dir/'failed_requests.jsonl'
        if self.failure_log.exists():
            self.previous_requests += sum(json.loads(line)['http_attempts']
                for line in self.failure_log.read_text(encoding='utf-8').splitlines() if line.strip())

    def _request(self, sample, questions):
        payload = {'model': self.model, 'state': sample.public_state(), 'questions': questions}
        signature = digest({'endpoint': self.endpoint, 'payload': payload})
        target = self.cache_dir / (signature + '.json')
        if target.exists():
            record = json.loads(target.read_text(encoding='utf-8'))
            if record['response'].get('model'):self.resolved_models.add(record['response']['model'])
            self.cache_hits += 1
            self.records.append(record)
            return record['response']['answers']
        if self.previous_requests + self.calls >= self.max_requests:
            raise RuntimeError('Live run request budget reached; results already checkpointed')
        # Actual usage is available only after a response; reserve conservatively.
        reserve = len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        if self.input_tokens + reserve > self.max_input_tokens:
            raise RuntimeError('Live run input-token budget reached; results already checkpointed')
        first = self.calls
        start = monotonic()
        self.last_response = None
        try:
            answers = super()._request(sample, questions)
        except Exception as exc:
            incident={'utc':datetime.now(timezone.utc).isoformat(),'sample_id':sample.id,
                      'request_sha256':signature,'http_attempts':self.calls-first,
                      'error':str(exc).replace(self.api_key,'[REDACTED]'),
                      'billing':'unknown; no completed response, excluded from token estimate'}
            with self.failure_log.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(incident,ensure_ascii=False)+'\n')
            raise
        response = self.last_response
        if response is None:
            raise RuntimeError('Missing complete API response for audit')
        if response.get('model'):self.resolved_models.add(response['model'])
        usage = response.get('usage', {})
        self.input_tokens += usage.get('input_tokens', 0)
        self.output_tokens += usage.get('output_tokens', 0)
        record = {'utc': datetime.now(timezone.utc).isoformat(), 'sample_id': sample.id,
                  'endpoint': self.endpoint, 'request_sha256': signature,
                  'request': payload, 'response': response,
                  'http_attempts': self.calls-first, 'elapsed_seconds': monotonic()-start}
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)
        self.records.append(record)
        return answers
