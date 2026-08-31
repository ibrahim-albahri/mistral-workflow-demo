"""Fields extracted from supported personal-document categories."""

PERSONAL_COMMON_FIELDS: list[tuple[str, str]] = [
    ("full_name", "Full legal name as shown on the document"),
    ("date_of_birth", "Date of birth if present"),
    ("document_number", "Document identifier or registration number"),
    ("issue_date", "Date the document was issued"),
    ("expiry_date", "Date the document expires, if applicable"),
    ("nationality", "Nationality or citizenship, if mentioned"),
    ("address", "Residential or mailing address, if present"),
]

PERSONAL_DOCUMENT_SPECIFIC_FIELDS: dict[str, list[tuple[str, str]]] = {
    "id": [
        ("issuing_authority", "Authority that issued the ID"),
        ("document_type", "Type of identification document"),
        ("place_of_birth", "Place of birth"),
        ("sex", "Sex or gender field if present"),
        ("mrz", "Machine-readable zone lines, transcribed exactly when visible"),
    ],
    "passport": [
        ("passport_number", "Passport number"),
        ("country_of_issue", "Country that issued the passport"),
        ("place_of_birth", "Place of birth"),
        ("passport_type", "Type of passport or travel document"),
        ("holder_signature", "Whether the holder signature is present"),
        ("mrz", "Machine-readable zone lines, transcribed exactly when visible"),
    ],
    "proof_of_address": [
        ("provider_name", "Name of the service provider or issuer"),
        ("provider_address", "Provider address if shown"),
        ("account_holder_name", "Account holder or customer name"),
        ("utility_type", "Type of utility or bill"),
        ("billing_period", "Billing or service period"),
    ],
    "gtc": [
        ("document_title", "Title or name of the document"),
        ("issuer_name", "Issuer or company name"),
        ("version_date", "Effective or version date"),
        ("key_clauses", "Main clauses or obligations described"),
        ("acceptance_text", "Any acceptance or acknowledgment wording"),
    ],
    "other": [],
}

PERSONAL_DOCUMENT_CATEGORIES: list[str] = list(PERSONAL_DOCUMENT_SPECIFIC_FIELDS.keys())

PERSONAL_DOCUMENT_LABELS: dict[str, str] = {
    "id": "🆔 ID",
    "passport": "🛂 Passport",
    "proof_of_address": "🏠 Proof of Address",
    "gtc": "📜 GTC",
    "other": "❓ Other",
}

# Fields with a value type other than the default optional string. The
# extraction workflow uses these values when it builds its Pydantic schema.
PERSONAL_FIELD_TYPES: dict[str, type] = {
    "holder_signature": bool,
    "key_clauses": list[str],
}
