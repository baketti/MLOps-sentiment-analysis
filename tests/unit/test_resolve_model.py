from unittest.mock import patch
from huggingface_hub.errors import RepositoryNotFoundError
from api.utils.utilities import resolve_model


def test_resolve_model_finetuned_available():
    with patch("api.utils.utilities.model_info"):
        result = resolve_model("base-model", "finetuned-model")
    assert result == "finetuned-model"


def test_resolve_model_fallback_to_base():
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.status_code = 404
    error = RepositoryNotFoundError("not found", response=mock_response)
    with patch("api.utils.utilities.model_info", side_effect=error):
        result = resolve_model("base-model", "finetuned-model")
    assert result == "base-model"
