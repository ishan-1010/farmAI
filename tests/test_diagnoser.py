import json
import pytest
from unittest.mock import patch, MagicMock
from agents.diagnoser import diagnose_disease, SEVERITY_LABELS

DUMMY_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 100

MOCK_DISEASED = {
    "disease_name": "Early Blight",
    "pathogen": "Alternaria solani",
    "severity": 3,
    "severity_label": "Moderate",
    "affected_area_percent": 35,
    "symptoms_observed": ["brown spots", "yellowing leaves"],
    "is_healthy": False,
}

MOCK_HEALTHY = {
    "disease_name": "None",
    "pathogen": "None",
    "severity": 0,
    "severity_label": "Healthy",
    "affected_area_percent": 0,
    "symptoms_observed": [],
    "is_healthy": True,
}


def _make_mock_client(response_dict):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(response_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestDiagnoseDisease:
    @patch("agents.diagnoser.Groq")
    def test_schema_all_keys_present(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_DISEASED)
        result = diagnose_disease(DUMMY_IMAGE, "Tomato")
        for key in ["problem_type", "disease_name", "pathogen", "severity", "severity_label", "affected_area_percent", "symptoms_observed", "is_healthy"]:
            assert key in result, f"Missing key: {key}"

    @patch("agents.diagnoser.Groq")
    def test_diseased_plant_detection(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_DISEASED)
        result = diagnose_disease(DUMMY_IMAGE, "Tomato")
        assert result["is_healthy"] is False
        assert result["disease_name"] == "Early Blight"
        assert result["severity"] == 3

    @patch("agents.diagnoser.Groq")
    def test_healthy_plant_path(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_HEALTHY)
        result = diagnose_disease(DUMMY_IMAGE, "Tomato")
        assert result["is_healthy"] is True
        assert result["disease_name"] == "None"
        assert result["severity"] == 0
        assert result["severity_label"] == "Healthy"

    @patch("agents.diagnoser.Groq")
    def test_severity_out_of_range_clamped(self, mock_groq):
        bad_response = {**MOCK_DISEASED, "severity": 99}
        mock_groq.return_value = _make_mock_client(bad_response)
        result = diagnose_disease(DUMMY_IMAGE, "Tomato")
        assert result["severity"] == 0  # reset to 0 when invalid

    @patch("agents.diagnoser.Groq")
    def test_severity_label_mapped_correctly(self, mock_groq):
        for sev, label in SEVERITY_LABELS.items():
            response = {**MOCK_DISEASED, "severity": sev, "severity_label": label}
            mock_groq.return_value = _make_mock_client(response)
            result = diagnose_disease(DUMMY_IMAGE, "Tomato")
            assert result["severity_label"] == SEVERITY_LABELS[result["severity"]]
