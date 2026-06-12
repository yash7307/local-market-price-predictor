# streamlit_app.py — Entry point for Streamlit Community Cloud
# Delegates to dashboard/app.py

import runpy
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

# Run the actual dashboard
runpy.run_path(
    str(Path(__file__).parent / "dashboard" / "app.py"),
    run_name="__main__",
)
