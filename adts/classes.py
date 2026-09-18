"""WALDO30 class list. The order must match the model's output indices. It's the same for
every WALDO30 checkpoint (n, n-p2, m, l, l-p2), and export_hailo.py checks it again
against the .pt before exporting. A compiled .hef carries no class names, so
HailoDetector takes them from here."""

WALDO_NAMES = [
    "LightVehicle", "Person", "Building", "UPole", "Boat", "Bike",
    "Container", "Truck", "Gastank", "Digger", "SolarPanels", "Bus",
]

# Turkish display names, taken from track.py (the user chose the translations).
# ASCII-transliterated on purpose: cv2.putText can't draw non-ASCII characters.
WALDO_NAMES_TR = {
    "LightVehicle": "Otomobil",
    "Person": "Insan",
    "Building": "Bina",
    "UPole": "Elektrik Diregi",
    "Boat": "Tekne",
    "Bike": "Bisiklet/Motosiklet",
    "Container": "Konteyner",
    "Truck": "Kamyon",
    "Gastank": "Gaz Tanki",
    "Digger": "Is Makinesi",
    "SolarPanels": "Gunes Paneli",
    "Bus": "Otobus",
}

# Fixed symbology labels, switched by the same "dil degistir" command as the class names.
# ASCII only, for the same reason as WALDO_NAMES_TR.
UI_TEXT = {
    "en": {"IDLE": "IDLE", "LOCKED": "LOCKED", "COAST": "COAST", "LOST": "LOST",
           "ai": "AI", "off": "OFF", "tgt": "TGT", "sel": "SEL", "gate": "GATE", "scene": "SCENE"},
    "tr": {"IDLE": "BOSTA", "LOCKED": "KILITLI", "COAST": "TAHMIN", "LOST": "KAYIP",
           "ai": "YZ", "off": "KAPALI", "tgt": "HEDEF", "sel": "SECILI", "gate": "KAPI", "scene": "SAHNE"},
}


def display_names(names, lang):
    return [WALDO_NAMES_TR.get(n, n) for n in names] if lang == "tr" else list(names)


def ui_text(lang):
    return UI_TEXT.get(lang, UI_TEXT["en"])
