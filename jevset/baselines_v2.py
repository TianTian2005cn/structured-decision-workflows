"""Direct Jev Choice thresholds and a first-token Choice Funnel baseline.

Choice Funnel control flow follows the authors' default configuration at
sata-bench/sata-bench@371dd0c (MIT). Position priors are fitted on development
calibration inputs only; no test-prefix calibration or oracle stopping is used.
"""
from __future__ import annotations
import math
from typing import Protocol
from .schema import Sample
from .backend import softmax


def direct_choice(backend,sample):
    """Normalized original-option scores, not binary inclusion probabilities."""
    labels=list(sample.options)
    if len(labels)>254:raise ValueError('Direct Choice with none needs <=254 labels')
    criteria={f'o{i}':f'{k}: {sample.options[k]}' for i,k in enumerate(labels)}
    criteria['none']='None of the listed options is correct or relevant.'
    answer=backend._request(sample,{'direct':{'type':'choice',
        'instructions':'Which ONE original option best answers the task? Choose none if no original option applies.',
        'criteria':criteria}})['direct']
    raw=backend._probabilities(answer,list(criteria))
    return {k:raw[f'o{i}'] for i,k in enumerate(labels)},raw['none']


def choice_threshold(scores,threshold):
    if not 0<=threshold<=1:raise ValueError('threshold outside [0,1]')
    return frozenset(k for k,v in scores.items() if v>=threshold)


class FirstTokenScorer(Protocol):
    def probabilities(self,sample:Sample,options:list[tuple[str,str]]) -> list[float]:...


def fit_position_priors(scorer:FirstTokenScorer,samples):
    """Unsupervised cyclic PriDe calibration on development inputs only."""
    priors=[];calls=0
    for sample in samples:
        if sample.split not in {'train','validation','development'}:
            raise ValueError('Choice Funnel priors cannot use held-out/test inputs')
        opts=list(sample.options.items());n=len(opts);observed=[]
        if not 1<n<=26:raise ValueError('First-token reference expects 2..26 options')
        for i in range(n):
            raw=scorer.probabilities(sample,opts[i:]+opts[:i]);calls+=1
            if len(raw)!=n or any(p<0 or not math.isfinite(p) for p in raw) or sum(raw)<=0:raise ValueError('Invalid first-token distribution')
            observed.append([p/sum(raw) for p in raw])
        # Authors use softmax(mean(log observed)) across display positions.
        dist=softmax({i:sum(math.log(row[i]+1e-10) for row in observed)/n for i in range(n)})
        priors.append([dist[i] for i in range(n)])
    collected={}
    for prior in priors:
        for n in range(2,len(prior)+1):
            prefix=prior[:n];z=sum(prefix)
            collected.setdefault(n,[]).append([v/z for v in prefix])
    return {'source':'DEVELOPMENT INPUTS ONLY','calls':calls,
            'by_option_count':{n:[sum(row[i] for row in rows)/len(rows) for i in range(n)] for n,rows in collected.items()}}


def choice_funnel(sample,scorer:FirstTokenScorer,priors,probability_limit=.9):
    """Author-default stop rules ported to typed callbacks.

    Original defaults force at least one pick, append None-of-the-above only
    after that pick, disable probability-drop stopping, and compare the current
    (possibly debiased log) score with 0.9 AFTER accepting later picks. The mixed
    score scale is retained for reference fidelity, not endorsed as calibration.
    A Jev Choice callback must be labeled a separate adaptation, never this
    first-token baseline. Empty-set tasks expose the forced-first-pick limit.
    """
    if priors.get('source')!='DEVELOPMENT INPUTS ONLY':raise ValueError('Unaudited position priors')
    remaining=list(sample.options.items());picked=[];trace=[];calls=0
    while remaining:
        opts=remaining+([('__none__','None of the above')] if picked else [])
        raw=scorer.probabilities(sample,opts);calls+=1
        if len(raw)!=len(opts) or sum(raw)<=0 or any(p<0 or not math.isfinite(p) for p in raw):raise ValueError('Invalid first-token scores')
        raw=[p/sum(raw) for p in raw]
        prior=priors['by_option_count'].get(len(opts),priors['by_option_count'].get(str(len(opts))))
        values=[math.log(p+1e-10)-math.log(prior[i]+1e-10) for i,p in enumerate(raw)] if prior else raw
        best=max(range(len(opts)),key=lambda i:(values[i],-i))
        key=opts[best][0]
        trace.append({'options':[k for k,_ in opts],'raw_probabilities':raw,'selected':key,
                      'selected_score':values[best],'debiased':prior is not None})
        if key=='__none__':break
        was_picked=bool(picked);picked.append(key)
        remaining=[kv for kv in remaining if kv[0]!=key]
        if not remaining:break
        if was_picked and values[best]<probability_limit:break
    return frozenset(picked),{'calls':calls,'trace':trace,'variant':'author-default first-token port',
        'source_commit':'371dd0c18fe75a96fbbcf2d1507ceeaf0d5263c5'}


class LocalHFFirstTokenScorer:
    """Optional real generative baseline, local model only; no automatic download.

    Requires torch+transformers. Explicit display-token prefix is required because
    option-letter tokenization differs by model. No remote model code is executed.
    """
    def __init__(self,model_path,*,token_prefix='',device='cpu'):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False).to(device).eval()
        self.tokens=[self.tokenizer.convert_tokens_to_ids(token_prefix+chr(65+i)) for i in range(26)]
        self.calls=0

    def probabilities(self,sample,options):
        if not 1<len(options)<=26:raise ValueError('Expected 2..26 first-token options')
        ids=self.tokens[:len(options)]
        if any(v is None or v==self.tokenizer.unk_token_id for v in ids) or len(set(ids))!=len(ids):
            raise ValueError('Display labels are not unique known tokens; set the correct --token-prefix')
        text=sample.text
        if '\nContext: ' in text and '\nQuestion: ' in text:
            paragraph,question=text.split('\nContext: ',1)[1].split('\nQuestion: ',1)
        else:paragraph=text;question='Select all correct options.'
        prompt=('You are presented with the following:\nParagraph: '+paragraph+'\nQuestion: '+question+
                '\nChoices:\n'+'\n'.join(f'{chr(65+i)}. {value}' for i,(_,value) in enumerate(options))+
                '\n\nTask:\nIdentify and select all the correct answers based on the paragraph and the question.\n\nAnswers:')
        inputs=self.tokenizer(prompt,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            logits=self.model(**inputs).logits[0,-1,ids]
            probs=self.torch.softmax(logits,dim=-1).cpu().tolist()
        self.calls+=1
        return probs
