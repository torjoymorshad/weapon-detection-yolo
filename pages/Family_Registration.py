import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from family_manager import family_dashboard
family_dashboard()
