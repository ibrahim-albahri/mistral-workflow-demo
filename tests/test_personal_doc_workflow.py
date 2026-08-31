from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.extraction_fields import (
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_LABELS,
)
from workflows.personal_doc_workflow import PersonalDocumentWorkflow


def test_personal_document_categories_are_defined():
    assert PERSONAL_DOCUMENT_CATEGORIES == [
        "id",
        "passport",
        "proof_of_address",
        "gtc",
    ]
    assert PERSONAL_DOCUMENT_LABELS["id"] == "🆔 ID"
    assert PERSONAL_DOCUMENT_LABELS["passport"] == "🛂 Passport"
    assert PERSONAL_DOCUMENT_LABELS["proof_of_address"] == "🏠 Proof of Address"
    assert PERSONAL_DOCUMENT_LABELS["gtc"] == "📜 GTC"


def test_personal_document_workflow_is_registered():
    assert PersonalDocumentWorkflow.__name__ == "PersonalDocumentWorkflow"
