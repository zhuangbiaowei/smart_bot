#!/usr/bin/env python3
"""
Echo test script for SmartBot script-type skill testing.

This script demonstrates:
1. Argument parsing and handling
2. JSON output formatting
3. Error handling
"""

import sys
import json
import time
from datetime import datetime

def main():
    start_time = time.time()

    # Collect all arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Build result
    result = {
        "success": True,
        "script": "echo.py",
        "timestamp": datetime.now().isoformat(),
        "args_received": args,
        "arg_count": len(args),
    }

    # Simulate some processing
    if args:
        result["output"] = f"Echo: {' '.join(args)}"
    else:
        result["output"] = "Echo: (no arguments)"

    # Calculate execution time
    result["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)

    # Output as JSON
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
