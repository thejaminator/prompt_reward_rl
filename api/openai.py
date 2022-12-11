import requests as requests
from pydantic import BaseModel

from settings import OPENAI_KEY


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


DEFAULT_AUTH_KEY = OPENAI_KEY


def get_moderations(text: str, auth_key: str = DEFAULT_AUTH_KEY) -> OpenAIModeration:
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
    json = request.json()
    # parse it into a nicer format
    id = json["id"]
    model = json["model"]
    hate_field = ModerationField(
        prediction=json["results"][0]["categories"]["hate"],
        proba_score=json["results"][0]["category_scores"]["hate"],
    )


def _parse_openai_field(raw_json: dict, field_name: str) -> ModerationField:
    return ModerationField(
        prediction=raw_json["results"][0]["categories"][field_name],
        proba_score=raw_json["results"][0]["category_scores"][field_name],
    )


def test_parse_openai_field():
    test_dict = {
        "id": "modr-6M9WCFPleGTp0299OAyNCcGieOqv3",
        "model": "text-moderation-004",
        "results": [
            {
                "categories": {
                    "hate": False,
                    "hate/threatening": False,
                    "self-harm": False,
                    "sexual": False,
                    "sexual/minors": False,
                    "violence": False,
                    "violence/graphic": False,
                },
                "category_scores": {
                    "hate": 3.7888854421908036e-05,
                    "hate/threatening": 2.24931300181197e-05,
                    "self-harm": 0.00020809986745007336,
                    "sexual": 2.9429097594402265e-06,
                    "sexual/minors": 4.750487789806357e-07,
                    "violence": 0.4000481367111206,
                    "violence/graphic": 0.006268431898206472,
                },
                "flagged": False,
            }
        ],
    }
    parsed: ModerationField = _parse_openai_field(test_dict, "hate")
    assert parsed.prediction is False
    assert parsed.proba_score == 3.7888854421908036e-05
