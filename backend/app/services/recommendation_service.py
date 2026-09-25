"""
SKINORA Backend — Recommendation Service

STATUS: Rule-based placeholder (Phase 1).

Provides general skincare guidance based on acne severity level.
These are NOT medical diagnoses or personalized prescriptions.
They are general educational guidelines clearly presented as such.

In Phase 3, this service will be updated to use real ML observations
as input to a more sophisticated recommendation engine.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# General rule-based recommendations by acne severity level
# ---------------------------------------------------------------------------

_ACNE_RECOMMENDATIONS: dict[int, dict] = {
    0: {
        "level": "Clear / Minimal",
        "description": "Skin shows no significant acne.",
        "guidance": [
            "Maintain a gentle twice-daily cleansing routine.",
            "Use non-comedogenic moisturiser appropriate for your skin type.",
            "Apply broad-spectrum SPF 30+ sunscreen daily.",
            "Avoid touching your face frequently.",
        ],
    },
    1: {
        "level": "Mild Acne",
        "description": "Small number of comedones or minor breakouts.",
        "guidance": [
            "Use a gentle, non-soap cleanser twice daily.",
            "Consider over-the-counter products containing salicylic acid or benzoyl peroxide.",
            "Avoid heavy, occlusive makeup — opt for non-comedogenic formulations.",
            "Consult a dermatologist if symptoms persist or worsen.",
        ],
    },
    2: {
        "level": "Moderate Acne",
        "description": "Multiple inflammatory lesions or papules/pustules.",
        "guidance": [
            "See a dermatologist for a personalised treatment plan.",
            "Prescription topicals (e.g., retinoids) are often effective at this stage.",
            "Avoid picking or squeezing lesions to reduce scarring risk.",
            "Keep skin hydrated — dryness can worsen inflammation.",
        ],
    },
    3: {
        "level": "Severe Acne",
        "description": "Extensive inflammation, nodules, or cystic acne.",
        "guidance": [
            "Seek professional dermatological care promptly.",
            "Systemic treatments (e.g., oral antibiotics, isotretinoin) may be indicated.",
            "Do not attempt to self-treat severe acne — this may worsen scarring.",
            "A dermatologist can tailor treatment to your skin's needs.",
        ],
    },
}

_GENERAL_GUIDANCE = {
    "description": "General skincare guidance (no analysis available yet).",
    "guidance": [
        "Cleanse gently twice daily with a product suited to your skin type.",
        "Moisturise daily with a non-comedogenic product.",
        "Apply SPF 30+ sunscreen every morning.",
        "Stay hydrated and maintain a balanced diet.",
        "Consult a qualified dermatologist for personalised skin advice.",
    ],
    "disclaimer": (
        "SKINORA provides general skincare information for educational purposes only. "
        "This is not medical advice. Always consult a licensed healthcare professional "
        "for diagnosis and treatment."
    ),
}


def get_recommendations(acne_severity: Optional[int] = None) -> dict:
    """Return rule-based skincare recommendations.

    Args:
        acne_severity: Acne severity level (0–3).  Pass None if
                       no analysis result is available yet.

    Returns:
        Dictionary with level, guidance list, and disclaimer.
    """
    disclaimer = (
        "SKINORA provides general skincare information for educational purposes only. "
        "This is not medical advice. Always consult a licensed healthcare professional "
        "for diagnosis and treatment."
    )

    if acne_severity is None:
        logger.debug("Recommendations requested with no severity — returning general guidance.")
        result = dict(_GENERAL_GUIDANCE)
        result["disclaimer"] = disclaimer
        return result

    if acne_severity not in _ACNE_RECOMMENDATIONS:
        logger.warning("Unknown acne_severity=%s — returning general guidance.", acne_severity)
        result = dict(_GENERAL_GUIDANCE)
        result["disclaimer"] = disclaimer
        return result

    rec = dict(_ACNE_RECOMMENDATIONS[acne_severity])
    rec["disclaimer"] = disclaimer
    logger.debug("Recommendations returned for severity=%s", acne_severity)
    return rec
