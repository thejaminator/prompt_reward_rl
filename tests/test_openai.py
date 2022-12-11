from api.openai import ModerationField, _parse_openai_field


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
