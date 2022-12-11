from api.openai import get_moderations, OpenAIModeration


def test_get_moderations():
    moderated: OpenAIModeration = get_moderations("lets kill someone")
    assert moderated.fields.violence.prediction is True
