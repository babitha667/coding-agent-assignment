# Prevent pytest from treating the imported test_failure_reason helper
# (from src.validation) as a test case. It is a utility function, not a test.
collect_ignore = []

def pytest_collect_file(parent, file_path):
    pass

def pytest_ignore_collect(collection_path, config):
    # Don't try to collect src source files
    if collection_path.suffix == '.py' and 'tests' not in str(collection_path):
        return True
