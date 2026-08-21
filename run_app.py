import sys
import os
from pathlib import Path

# Fix sys.path for PyInstaller bundled environment
if getattr(sys, 'frozen', False):
    base_dir = Path(sys._MEIPASS)
else:
    base_dir = Path(__file__).resolve().parent

sys.path.insert(0, str(base_dir))

import streamlit.web.cli as stcli

if __name__ == "__main__":
    app_path = str(base_dir / "app.py")
    sys.argv = ["streamlit", "run", app_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())
