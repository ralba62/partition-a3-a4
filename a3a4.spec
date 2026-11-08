# -*- mode: python ; coding: utf-8 -*-
# Configuration PyInstaller : une seule recette pour Mac et Windows.
import os
import sys
from PyInstaller.utils.hooks import collect_submodules

NOM = "Partition A3 vers A4"
datas, hidden = [], []

if sys.platform == "darwin":
    # lecture des numéros de page : OCR intégré à macOS (Vision), via pyobjc
    for m in ("objc", "Foundation", "Quartz", "CoreML", "Vision", "AppKit"):
        try:
            hidden += collect_submodules(m)
        except Exception:
            pass

if os.path.isdir("vendor/tesseract"):
    # Windows : Tesseract est copié ici par la recette GitHub
    datas.append(("vendor/tesseract", "tesseract"))

a = Analysis(
    ["a3_to_a4.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["scipy", "matplotlib", "pandas", "PyQt5", "PyQt6", "PySide6", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NOM,
    debug=False,
    strip=False,
    upx=False,
    console=os.environ.get("A3A4_CONSOLE") == "1",   # fenêtre noire seulement pour les essais
    icon="icon.ico" if sys.platform.startswith("win") else None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NOM)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=NOM + ".app",
        icon="icon.icns",
        bundle_identifier="fr.partitions.a3versa4",
        info_plist={
            "CFBundleName": NOM,
            "CFBundleDisplayName": NOM,
            "CFBundleShortVersionString": "1.0",
            "NSHighResolutionCapable": True,
            "NSDocumentsFolderUsageDescription": "Lire vos partitions PDF et enregistrer les versions A4.",
            "NSDesktopFolderUsageDescription": "Lire vos partitions PDF et enregistrer les versions A4.",
            "NSDownloadsFolderUsageDescription": "Lire vos partitions PDF et enregistrer les versions A4.",
        },
    )
