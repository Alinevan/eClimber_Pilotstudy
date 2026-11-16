"""Stable labels used when interpreting the questionnaire template."""

BODY_WEIGHT_OPTIONS = ("25%", "50%", "75%", "None")

B1_OPTIONS = (
    "zu weit weg, mein Klettern wurde gestört",
    "mit einer angenehmen Distanz",
    "zu nahe, mein Klettern wurde gestört",
)

YES_NO_UNSURE = ("Ja", "Nein", "Unsicher")

D2_OPTIONS = (
    "Die Sicherungsleine zum Ausruhen stoppen, bis ich das System wieder aktiviere.",
    "Die Sicherungsleine zum Ausruhen stoppen und das System nach einer vordefinierten Zeit wieder aktivieren.",
    "Nur die Gewichtsunterstützung stoppen, damit ich ohne Unterstützung gesichert weiterklettern kann oder beim Loslassen, kontrolliert abgelassen werde.",
    "Andere",
)

TABLE_CONDITION_QUESTIONS = {"A1", "A2", "A3", "A4", "A5"}
TABLE_QUESTIONS = {"C1", "C2", "E1"}

# Used only for stable human-readable output. The source DOCX stores some
# answers in tables and others in paragraphs, so source order alone is not
# the questionnaire order.
QUESTION_ORDER = {
    "A1": 10, "A2": 20, "A3": 30, "A4": 40, "A5": 50, "A6": 60,
    "A_Bemerkungen": 65,
    "B1": 70, "B_Bemerkungen": 75,
    "C1": 80, "C2": 90, "C_Bemerkungen": 95,
    "D1": 100, "D2": 110, "D_Bemerkungen": 115,
    "E1": 120, "E2": 130, "E2_Bedürfnis": 135, "E_Bemerkungen": 138,
    "F1": 140, "F2": 150, "F3": 160, "F_Bemerkungen": 165,
    "Generelle Bemerkungen": 170,
}
