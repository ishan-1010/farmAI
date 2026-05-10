import json
import pytest
from unittest.mock import patch, MagicMock
from agents.identifier import identify_crop, encode_image

DUMMY_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG bytes

MOCK_RESPONSE = {
    "crop_name": "Tomato",
    "scientific_name": "Solanum lycopersicum",
    "confidence": "high",
    "notes": "Characteristic serrated leaflets visible",
}


def _make_mock_client(response_dict):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(response_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestEncodeImage:
    def test_returns_string(self):
        result = encode_image(DUMMY_IMAGE)
        assert isinstance(result, str)

    def test_base64_decodable(self):
        import base64
        result = encode_image(DUMMY_IMAGE)
        decoded = base64.standard_b64decode(result)
        assert decoded == DUMMY_IMAGE


class TestIdentifyCrop:
    @patch("agents.identifier.Groq")
    def test_returns_correct_schema(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_RESPONSE)
        result = identify_crop(DUMMY_IMAGE)
        assert "crop_name" in result
        assert "scientific_name" in result
        assert "confidence" in result
        assert "notes" in result

    @patch("agents.identifier.Groq")
    def test_values_match_mock(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_RESPONSE)
        result = identify_crop(DUMMY_IMAGE)
        assert result["crop_name"] == "Tomato"
        assert result["confidence"] == "high"

    @patch("agents.identifier.Groq")
    def test_unknown_crop_handled(self, mock_groq):
        unknown_response = {"crop_name": "Unknown", "scientific_name": "Unknown", "confidence": "low", "notes": ""}
        mock_groq.return_value = _make_mock_client(unknown_response)
        result = identify_crop(DUMMY_IMAGE)
        assert result["crop_name"] == "Unknown"
        assert result["confidence"] == "low"

    @patch("agents.identifier.Groq")
    def test_missing_keys_get_defaults(self, mock_groq):
        # LLM returns partial response
        mock_groq.return_value = _make_mock_client({"crop_name": "Wheat"})
        result = identify_crop(DUMMY_IMAGE)
        assert result["crop_name"] == "Wheat"
        assert result["scientific_name"] == "Unknown"
        assert result["confidence"] == "low"
        assert result["notes"] == ""
