"""Check actual production objects for direct allocator references."""
import re
import subprocess
import sys
for path in sys.argv[1:]:
    symbols=subprocess.check_output(['nm','-u',path],text=True)
    if re.search(r'\b_?(malloc|calloc|realloc|aligned_alloc|free)\b',symbols):
        raise SystemExit('FAIL: allocator reference in '+path)
print('PASS: production graph/planner/executor objects have no direct allocator references')
