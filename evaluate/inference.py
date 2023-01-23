from enum import Enum
from typing import Optional, Dict, Any, List, Union

import numpy as np
import openai
from openai import APIError
from pydantic import BaseModel, conlist
from slist import Slist
from slist.pydantic_compat import SlistPydantic

from api.redis_cache import redis_cache


class OpenaiInferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: int
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Union[None, str, conlist(str, min_items=1, max_items=4)] = None  # type: ignore


class TokenProba(BaseModel):
    token: str
    log_prob: float


class TokenInfo(BaseModel):
    token: str  # this is the token that got sampled
    log_prob: float  # the first token in the prompt will always have a log_prob of 0.0
    text_offset: int  # the offset of the token in the text
    # the top 5 tokens in the probability distribution
    # for first token in the prompt this is empty
    top_5_tokens: SlistPydantic[TokenProba]


class FinishReasons(str, Enum):
    stop = "stop"
    length = "length"


class GPTFullResponse(BaseModel):
    id: Optional[str]
    prompt: str
    completion: str
    prompt_token_infos: SlistPydantic[TokenInfo]
    completion_token_infos: SlistPydantic[TokenInfo]
    # so we can view this in mongo easily
    completion_total_log_prob: float
    average_completion_total_log_prob: Optional[float]
    finish_reason: FinishReasons

    @property
    def token_infos(self) -> SlistPydantic[TokenInfo]:
        return self.prompt_token_infos + self.completion_token_infos

    @property
    def completion_tokens_length(self) -> int:
        return len(self.completion_token_infos)

    @property
    def average_completion_prob(self) -> Optional[float]:
        completion_token_infos_log_prob: Slist[float] = self.completion_token_infos.map(
            lambda token_info: token_info.log_prob
        )
        # convert them into probabilities and then average them
        probas: Slist[float] = completion_token_infos_log_prob.map(
            lambda log_prob: np.exp(log_prob)
        )
        return probas.average()


def parse_gpt_response(
    prompt: str, response_dict: Dict[Any, Any], end_tokens: set[str]
) -> GPTFullResponse:
    response_id = response_dict["id"]
    completion = response_dict["choices"][0]["text"][len(prompt) :]
    logprobs: List[Union[int, None]] = response_dict["choices"][0]["logprobs"][
        "token_logprobs"
    ]
    # the first token has a logprob of "None" so we need to change it to 0
    edited_logprobs: Slist[int] = Slist(logprobs).map(
        lambda x: x if x is not None else 0
    )
    tokens: Slist[str] = Slist(response_dict["choices"][0]["logprobs"]["tokens"])
    top_5_probabilities: Slist[Slist[TokenProba]] = Slist(
        response_dict["choices"][0]["logprobs"]["top_logprobs"]
    ).map(
        lambda maybe_dict: Slist.from_dict(maybe_dict).map(
            lambda tup: TokenProba(token=tup[0], log_prob=tup[1])
        )
        # the first token has None instead of a dict
        if maybe_dict is not None
        else Slist()
    )

    finish_reason = response_dict["choices"][0]["finish_reason"]
    offsets: Slist[int] = Slist(response_dict["choices"][0]["logprobs"]["text_offset"])

    token_infos: Slist[TokenInfo] = tokens.zip(
        edited_logprobs, top_5_probabilities, offsets
    ).map(
        lambda tup: TokenInfo(
            token=tup[0], log_prob=tup[1], top_5_tokens=tup[2], text_offset=tup[3]
        )
    )

    # now you need to find out where the prompt ends and the completion begins
    # using the text_offset
    prompt_offset = len(prompt)
    prompt_token_infos, completion_token_infos = token_infos.split_by(
        lambda x: x.text_offset < prompt_offset
    )
    # this is dumb, but sometimes openai adds tokens beyond the end token
    completion_token_infos = completion_token_infos.take_until_inclusive(
        lambda x: x.token in end_tokens
    )

    completion_token_infos_log_prob = completion_token_infos.map(
        lambda token_info: token_info.log_prob
    )

    return GPTFullResponse(
        id=response_id,
        prompt=prompt,
        completion=completion,
        prompt_token_infos=prompt_token_infos,
        completion_token_infos=completion_token_infos,
        completion_total_log_prob=completion_token_infos_log_prob.sum(),
        average_completion_total_log_prob=completion_token_infos_log_prob.average(),
        finish_reason=finish_reason,
    )


@redis_cache(decode_dict=GPTFullResponse)
def cached_get_openai_completion(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> GPTFullResponse:
    return get_openai_completion(config, prompt)


def get_openai_completion(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> GPTFullResponse:
    try:
        response_dict = openai.Completion.create(
            model=config.model,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            top_p=1,
            n=1,
            stream=False,
            stop=[config.stop] if isinstance(config.stop, str) else config.stop,
            # needed to get logprobs
            logprobs=5,
            # needed to get logprobs of prompt
            echo=True,
        )
    except APIError as e:
        print(f"APIError with prompt: {prompt}")
        raise e

    end_tokens: set[str] = (
        set(config.stop)
        if isinstance(config.stop, list)
        else {config.stop}
        if isinstance(config.stop, str)
        else set()
    )
    return parse_gpt_response(
        prompt=prompt, response_dict=response_dict, end_tokens=end_tokens
    )
