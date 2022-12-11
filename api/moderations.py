from typing import Any

import requests as requests
from pydantic import BaseModel
from retry import retry

from settings import DEFAULT_OPENAI_KEY


class ModerationField(BaseModel):
    """
    "hate": false,
    "hate/threatening": false,
    "self-harm": false,
    "sexual": false,
    "sexual/minors": false,
    "violence": false,
    "violence/graphic": false
    """

    prediction: bool
    proba_score: float


class ModerationFields(BaseModel):
    hate: ModerationField
    hate_threatening: ModerationField
    self_harm: ModerationField
    sexual: ModerationField
    sexual_minors: ModerationField
    violence: ModerationField
    violence_graphic: ModerationField


class OpenAIModeration(BaseModel):
    """
    See https://beta.openai.com/docs/guides/moderation/quickstart
    E.g.
        {
      "id": "modr-XXXXX",
      "model": "text-moderation-001",
      "results": [
        {
          "categories": {
            "hate": false,
            "hate/threatening": false,
            "self-harm": false,
            "sexual": false,
            "sexual/minors": false,
            "violence": false,
            "violence/graphic": false
          },
          "category_scores": {
            "hate": 0.18805529177188873,
            "hate/threatening": 0.0001250059431185946,
            "self-harm": 0.0003706029092427343,
            "sexual": 0.0008735615410842001,
            "sexual/minors": 0.0007470346172340214,
            "violence": 0.0041268812492489815,
            "violence/graphic": 0.00023186142789199948
          },
          "flagged": false
        }
      ]
    }
    """

    id: str
    model: str
    fields: ModerationFields


class RateLimitError(Exception):
    pass


def get_moderations(text: str, auth_key: str = DEFAULT_OPENAI_KEY) -> OpenAIModeration:
    """
    e.g.
    curl https://api.openai.com/v1/moderations \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"input": "Sample text goes here"}'
    """
    request = requests.post(
        "https://api.openai.com/v1/moderations",
        headers={"Authorization": f"Bearer {auth_key}"},
        json={"input": text},
    )
    rcode = request.status_code
    json = request.json()
    if rcode == 429:
        error_data = json["error"]
        raise RateLimitError(error_data.get("message"))
    # parse it into a nicer format
    id = json["id"]
    model = json["model"]
    hate_field = _parse_openai_field(json, "hate")
    hate_threatening_field = _parse_openai_field(json, "hate/threatening")
    self_harm_field = _parse_openai_field(json, "self-harm")
    sexual_field = _parse_openai_field(json, "sexual")
    sexual_minors_field = _parse_openai_field(json, "sexual/minors")
    violence_field = _parse_openai_field(json, "violence")
    violence_graphic_field = _parse_openai_field(json, "violence/graphic")

    fields = ModerationFields(
        hate=hate_field,
        hate_threatening=hate_threatening_field,
        self_harm=self_harm_field,
        sexual=sexual_field,
        sexual_minors=sexual_minors_field,
        violence=violence_field,
        violence_graphic=violence_graphic_field,
    )
    return OpenAIModeration(id=id, model=model, fields=fields)


@retry(tries=6, delay=10, backoff=2, exceptions=RateLimitError)
def get_moderations_retry(
    text: str, auth_key: str = DEFAULT_OPENAI_KEY
) -> OpenAIModeration:
    print("Done")
    return get_moderations(text, auth_key)


def _parse_openai_field(raw_json: dict[str, Any], field_name: str) -> ModerationField:
    return ModerationField(
        prediction=raw_json["results"][0]["categories"][field_name],
        proba_score=raw_json["results"][0]["category_scores"][field_name],
    )
