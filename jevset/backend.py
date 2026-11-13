"""Backends for decision primitives. Real TypeSafe endpoint per docs.typesafe.ai/api/system-one."""
from __future__ import annotations
from abc import ABC, abstractmethod
import json, math, os, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from http.client import HTTPException
from .schema import Sample


def clamp(p: float) -> float:
    return max(1e-8, min(1 - 1e-8, float(p)))


def softmax(logits: dict) -> dict:
    if not logits:
        raise ValueError('Empty logits')
    mx = max(logits.values())
    z = sum(math.exp(x - mx) for x in logits.values())
    return {k: math.exp(x - mx) / z for k, x in logits.items()}


class DecisionBackend(ABC):
    calls: int = 0
    @abstractmethod
    def judge(self, sample: Sample, *, cardinality: bool=True) -> tuple[dict[str,float], dict[int,float]]: ...
    @abstractmethod
    def choose(self, sample: Sample, candidates: list[frozenset[str]]) -> dict[int,float]: ...
    @abstractmethod
    def conditional_noul(self, sample: Sample, option: str) -> float: ...


class MockBackend(DecisionBackend):
    """Deterministic toy oracle using *mock_scores*, not gold; NOT a model or Jev measurement."""
    def __init__(self):
        self.calls = 0

    def judge(self, sample: Sample, *, cardinality: bool=True):
        self.calls += 1
        scores = {k: float(sample.mock_scores.get(k, .5)) for k in sample.options}
        if not cardinality:
            return scores, {}
        center = sample.mock_cardinality
        if center is None:
            center = round(sum(scores.values()))
        return scores, softmax({k: -0.8 * (k-center)**2 for k in range(len(scores)+1)})

    def choose(self, sample: Sample, candidates: list[frozenset[str]]):
        if not candidates:
            raise ValueError('No candidates')
        if len(candidates) > 255:
            raise ValueError('Choice limit exceeded')
        self.calls += 1
        def utility(s):
            score = sum(math.log(clamp(sample.mock_scores.get(k, .5))) if k in s
                        else math.log(1-clamp(sample.mock_scores.get(k, .5))) for k in sample.options)
            for pair, weight in sample.pair_weights.items():
                a,b = pair.split('|')
                if a in s and b in s:
                    score += weight
            return score
        return softmax({i: utility(s) for i, s in enumerate(candidates)})

    def conditional_noul(self, sample: Sample, option: str):
        self.calls += 1
        p = clamp(sample.mock_scores.get(option, .5))
        logit = math.log(p/(1-p))
        for prev in sample.selected_so_far:
            k = '|'.join(sorted([prev, option]))
            logit += sample.pair_weights.get(k, 0)
        return 1/(1+math.exp(-max(-30,min(30,logit))))


class JevBackend(DecisionBackend):
    """Real TypeSafe System One HTTP adapter. Requires TYPESAFE_API_KEY; not used by mock tests."""
    ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
    def __init__(self, *, api_key: str | None=None, model: str='jev-latest',
                 endpoint: str | None=None, timeout: float=45., retries: int=2):
        self.api_key = api_key or os.environ.get('TYPESAFE_API_KEY')
        if not self.api_key:
            raise ValueError('Set TYPESAFE_API_KEY (the official TypeSafe key), or use --backend mock')
        self.model = model
        self.endpoint = endpoint or self.ENDPOINT
        self.timeout = timeout
        self.retries = retries
        self.calls = 0

    def _request(self, sample: Sample, questions: dict) -> dict:
        payload = {'model':self.model, 'state':sample.public_state(), 'questions':questions}
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = Request(self.endpoint, data=body, headers={
            'Authorization': f'Bearer {self.api_key}', 'Content-Type':'application/json',
            'Accept':'application/json'}, method='POST')
        for attempt in range(self.retries+1):
            self.calls += 1
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                if 'answers' not in result or not isinstance(result['answers'], dict):
                    raise ValueError('TypeSafe response does not contain answers')
                self.last_response = result
                return result['answers']
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504, 529) or attempt == self.retries:
                    detail = exc.read(1000).decode('utf-8', 'replace')
                    raise RuntimeError(f'TypeSafe HTTP {exc.code}: {detail.replace(self.api_key, "[REDACTED]")}') from exc
            except (URLError, TimeoutError, HTTPException, ConnectionError):
                if attempt == self.retries:
                    raise
            time.sleep(min(2**attempt, 4))
        raise AssertionError('unreachable')

    @staticmethod
    def _probabilities(ans: dict, keys: list[str]) -> dict[str,float]:
        raw = ans.get('probabilities')
        if not isinstance(raw, dict):
            # Resilient to winning choice only, but do not misinterpret confidence as a distribution.
            raise ValueError('Choice response lacks probability distribution')
        vals = {k:float(raw.get(k, 0.0)) for k in keys}
        if any(v < 0 or not math.isfinite(v) for v in vals.values()) or sum(vals.values()) <= 0:
            raise ValueError('Invalid Choice probabilities')
        return {k:v/sum(vals.values()) for k,v in vals.items()}

    def judge(self, sample: Sample, *, cardinality: bool=True):
        keys = list(sample.options)
        questions = {f'label_{i}': {'type':'noul',
             'instructions':f'Is option {k} ({sample.options[k]}) a correct / relevant member of the required answer set?'}
             for i,k in enumerate(keys)}
        if cardinality and len(keys) <= 254:
            questions['cardinality'] = {'type':'choice',
              'instructions':'How many distinct options exactly should be selected in total? Select one count.',
              'criteria': {f'k{i}':f'Exactly {i} options are required' for i in range(len(keys)+1)}}
        # Conservative client chunking; the provider is limited by context size,
        # not a separate fixed number of questions (docs checked 2026-10-30).
        answer = {}
        parts = list(questions.items())
        for i in range(0,len(parts), 200):
            answer.update(self._request(sample, dict(parts[i:i+200])))
        scores = {}
        for i,k in enumerate(keys):
            entry = answer[f'label_{i}']
            scores[k] = float(entry['noul'])
            if not 0 <= scores[k] <= 1:
                raise ValueError('Noul outside [0,1]')
        cardinality_probs = {}
        if 'cardinality' in answer:
            cp = self._probabilities(answer['cardinality'], [f'k{i}' for i in range(len(keys)+1)])
            cardinality_probs = {i:cp[f'k{i}'] for i in range(len(keys)+1)}
        return scores, cardinality_probs

    def choose(self, sample: Sample, candidates: list[frozenset[str]]):
        if not 2 <= len(candidates) <= 255:
            if len(candidates)==1:
                return {0:1.0}
            raise ValueError('TypeSafe Choice expects 2-255 candidate sets')
        criteria = {f's{i}': ('Select exactly: '+ ', '.join(f'{k} ({sample.options[k]})' for k in sorted(group))
                    if group else 'Select NO options (empty answer set)')
                    for i,group in enumerate(candidates)}
        answer = self._request(sample, {'subset':{'type':'choice',
                      'instructions':'Which ONE of the following complete answer sets is most correct? All omitted options are excluded.',
                      'criteria':criteria}})['subset']
        raw = self._probabilities(answer, [f's{i}' for i in range(len(candidates))])
        return {i:raw[f's{i}'] for i in range(len(candidates))}

    def conditional_noul(self, sample: Sample, option: str):
        entry = self._request(sample, {'q': {'type':'noul',
            'instructions':f'Given the previously selected options in state, should option {option} ({sample.options[option]}) ALSO be selected?'}})['q']
        return float(entry['noul'])
