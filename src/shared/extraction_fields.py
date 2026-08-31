"""
Fields to extract by document type.
Imported by workflow.py and app.py.
"""

COMMON_FIELDS: list[tuple[str, str]] = [
    ("full_name", "Full name of the patient (last name + first name)"),
    ("patient_address", "Full address of the patient"),
    ("social_security_number", "Social security number of the patient"),
]

SPECIFIC_FIELDS: dict[str, list[tuple[str, str]]] = {
    "prescription": [
        ("doctor_name", "Name of the doctor who made the prescription"),
        ("doctor_address", "Address of the doctor who made the prescription"),
        ("doctor_rpps", "RPPS number of the doctor"),
        ("medications", "List of prescribed medications (separated by commas)"),
        ("prescription_date", "Date of the prescription"),
    ],
    "medical_bill": [
        ("healthcare_professional_name", "Name of the healthcare professional"),
        ("healthcare_professional_address", "Address of the healthcare professional"),
        ("bill_amount", "Total billed amount (with currency)"),
        ("services", "Billed services or procedures"),
        ("care_date", "Date of care"),
    ],
    "hospitalization_report": [
        ("hospital_name", "Name of the hospital"),
        ("department", "Department or care unit"),
        ("responsible_doctor", "Responsible doctor"),
        ("admission_date", "Date of admission"),
        ("discharge_date", "Date of discharge"),
        ("primary_diagnosis", "Primary diagnosis"),
    ],
    "biological_analysis": [
        ("laboratory", "Name of the laboratory"),
        ("prescribing_doctor", "Prescribing doctor"),
        ("sample_date", "Date of the sample"),
        ("analyses", "Performed analyses"),
        ("abnormal_results", "Reported abnormal results"),
    ],
    "medical_imaging": [
        ("facility", "Name of the imaging facility"),
        ("radiologist_name", "Name of the radiologist"),
        ("exam_type", "Type of exam (MRI, CT, X-ray, etc.)"),
        ("anatomical_region", "Anatomical region examined"),
        ("exam_date", "Date of the exam"),
        ("conclusion", "Conclusion or main finding"),
    ],
    "medical_certificate": [
        ("doctor_name", "Name of the signing doctor"),
        ("doctor_address", "Address of the doctor"),
        ("certificate_subject", "Subject of the certificate"),
        ("certificate_date", "Date of the certificate"),
        ("sick_leave_duration", "Duration of sick leave if mentioned"),
    ],
    "mutual_reimbursement": [
        ("mutual_name", "Name of the mutual insurance"),
        ("member_number", "Member number"),
        ("reimbursement_amount", "Amount reimbursed by the mutual insurance"),
        ("reimbursement_date", "Reimbursement date"),
        ("reimbursed_acts", "Reimbursed acts"),
    ],
    "social_security_reimbursement": [
        ("fund_name", "Name of the social security fund"),
        ("reimbursement_amount", "Amount reimbursed by social security"),
        ("reimbursement_rate", "Applied reimbursement rate"),
        ("reimbursement_date", "Reimbursement date"),
        ("reimbursed_acts", "Reimbursed acts"),
    ],
    "consultation_report": [
        ("doctor_name", "Name of the consulting doctor"),
        ("doctor_specialty", "Specialty of the doctor"),
        ("consultation_date", "Date of the consultation"),
        ("reason", "Reason for the consultation"),
        ("diagnosis", "Diagnosis"),
        ("prescribed_treatment", "Prescribed treatment or follow-up"),
    ],
    "informed_consent": [
        ("doctor_name", "Name of the doctor"),
        ("facility", "Facility"),
        ("medical_procedure", "Medical procedure concerned"),
        ("signature_date", "Date of signature"),
    ],
    "other": [],
}


DOCUMENT_CATEGORIES: list[str] = list(SPECIFIC_FIELDS.keys())

CATEGORY_LABELS: dict[str, str] = {
    "prescription": "📋 Prescription",
    "medical_bill": "🧾 Medical Bill",
    "hospitalization_report": "🏥 Hospitalization Report",
    "biological_analysis": "🔬 Biological Analysis",
    "medical_imaging": "🩻 Medical Imaging",
    "medical_certificate": "📄 Medical Certificate",
    "mutual_reimbursement": "💳 Mutual Reimbursement",
    "social_security_reimbursement": "🏛️ Social Security Reimbursement",
    "consultation_report": "👨‍⚕️ Consultation Report",
    "informed_consent": "✍️ Informed Consent",
    "other": "❓ Other",
}