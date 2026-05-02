"""
download_sample.py — Download sample datasheets for testing
Run this if you don't have your own PDFs yet.
"""

import urllib.request
import sys
from pathlib import Path

DATASHEETS_DIR = Path(__file__).parent.parent / "datasheets"

SAMPLES = [
    {
        "name": "esp32_datasheet.pdf",
        "url":  "https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf",
        "desc": "ESP32 SoC Datasheet (Espressif)",
    },
    # Fallback: a simpler public PDF if Espressif CDN is slow
    {
        "name": "atmega328p_datasheet.pdf",
        "url":  "https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf",
        "desc": "ATmega328P Datasheet (Microchip)",
    },
]


def download(url: str, dest: Path) -> bool:
    print(f"  Downloading {dest.name} …")
    print(f"  URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := resp.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {pct}% ({downloaded // 1024} KB / {total // 1024} KB)", end="", flush=True)
        print(f"\n  ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def create_minimal_test_pdf():
    """Create a clean synthetic 'datasheet' PDF using reportlab."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "reportlab",
                        "--break-system-packages", "--quiet"])
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

    dest = DATASHEETS_DIR / "test_component_datasheet.pdf"
    if dest.exists():
        print(f"  ✓ {dest.name} already exists, skipping.")
        return

    pages = [
        {
            "title": "TEST COMPONENT XC-9001 — DATASHEET",
            "lines": [
                "Manufacturer: HardwareCo Inc.",
                "Document Version: 1.2  |  Date: 2024-01",
                "",
                "1. ABSOLUTE MAXIMUM RATINGS",
                "   Supply Voltage (VCC):    3.3 V to 5.5 V",
                "   Absolute Max Voltage:    6.0 V (do not exceed)",
                "   Maximum Supply Current:  500 mA",
                "   Operating Temperature:   -40 °C to +85 °C",
                "   Storage Temperature:     -65 °C to +150 °C",
                "   ESD Protection:          2 kV HBM",
                "",
                "2. ELECTRICAL CHARACTERISTICS (VCC = 3.3 V, T = 25 °C)",
                "   Quiescent Current (active):   12 mA typical",
                "   Quiescent Current (sleep):    5 µA typical",
                "   Deep Sleep Current:           2.5 µA typical",
                "   Power-on Reset Threshold:     1.8 V",
            ],
        },
        {
            "title": "XC-9001 — DIGITAL I/O & COMMUNICATION INTERFACES",
            "lines": [
                "3. GPIO ELECTRICAL CHARACTERISTICS",
                "   Output High Voltage (VOH):    2.4 V min  @ IOH = -8 mA",
                "   Output Low Voltage (VOL):     0.4 V max  @ IOL =  8 mA",
                "   Input High Threshold (VIH):   0.7 × VCC",
                "   Input Low Threshold (VIL):    0.3 × VCC",
                "   Input Leakage Current:        ±1 µA max",
                "   Internal Pull-up Resistor:    40 kΩ typical",
                "   Maximum GPIO Sink Current:    25 mA per pin",
                "",
                "4. COMMUNICATION INTERFACES",
                "   SPI:  up to 40 MHz, modes 0–3, full-duplex",
                "   I²C:  up to 400 kHz (Fast Mode), 7-bit and 10-bit addressing",
                "   UART: up to 5 Mbps, configurable parity and stop bits",
                "   CAN:  2.0B Active, up to 1 Mbps",
                "   USB:  Full-Speed 12 Mbps (Device mode)",
                "",
                "5. FLASH MEMORY",
                "   Internal Flash:  512 KB program memory",
                "   EEPROM:          16 KB (emulated)",
                "   External SPI Flash: up to 128 MB supported",
            ],
        },
        {
            "title": "XC-9001 — PACKAGE & PINOUT",
            "lines": [
                "6. PACKAGE INFORMATION",
                "   Package:    QFN-48 (7 × 7 mm, 0.5 mm pitch)",
                "   Exposed Pad: GND (must be soldered for thermal dissipation)",
                "   Thermal Resistance (θJA): 28 °C/W",
                "   Thermal Resistance (θJC):  4 °C/W",
                "   Max Junction Temperature: 150 °C",
                "",
                "7. RECOMMENDED OPERATING CONDITIONS",
                "   VCC Supply:       3.0 V to 3.6 V (3.3 V nominal)",
                "   Core Voltage:     1.8 V (internal LDO regulated)",
                "   Clock Frequency:  up to 240 MHz",
                "   ADC Reference:    VCC or external 1.1 V – 3.3 V",
                "",
                "8. ORDERING INFORMATION",
                "   XC-9001-QFN48-T   Tape & Reel  2500 pcs",
                "   XC-9001-QFN48-B   Bulk tray    250 pcs",
                "",
                "For further information visit: www.hardwareco-example.com",
            ],
        },
    ]

    c = canvas.Canvas(str(dest), pagesize=letter)
    W, H = letter
    for pg in pages:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, H - 60, pg["title"])
        c.setFont("Helvetica", 11)
        y = H - 95
        for line in pg["lines"]:
            c.drawString(50, y, line)
            y -= 18
            if y < 60:
                break
        c.showPage()

    c.save()
    print(f"  ✓ Created synthetic test PDF: {dest.name}")


def main():
    DATASHEETS_DIR.mkdir(exist_ok=True)
    existing = list(DATASHEETS_DIR.glob("*.pdf"))
    if existing:
        print(f"[i] Already have {len(existing)} PDF(s) in datasheets/:")
        for p in existing:
            print(f"    {p.name}")
        print("    Delete them or add more to re-download.\n")
        return

    print("[+] Downloading sample datasheets …\n")
    success = False
    for s in SAMPLES:
        print(f"  [{s['desc']}]")
        dest = DATASHEETS_DIR / s["name"]
        if download(s["url"], dest):
            success = True
            break
        print()

    if not success:
        print("\n[!] Network downloads failed. Creating a synthetic test PDF instead …")
        create_minimal_test_pdf()

    print("\n[✓] Ready! Now run:  python src/ingest.py")


if __name__ == "__main__":
    main()
