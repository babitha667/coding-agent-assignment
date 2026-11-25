# Prevent pytest from collecting source modules that happen to have functions
# starting with "test_" (e.g., validation.py:test_failure_reason).
collect_ignore = ["src/validation.py"]
