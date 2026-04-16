#!/bin/bash

set -e

echo "=========================================="
echo "Image Classification API Test Suite"
echo "=========================================="

REPORT_DIR="test_reports"
mkdir -p $REPORT_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HTML_REPORT="${REPORT_DIR}/test_report_${TIMESTAMP}.html"
JUNIT_REPORT="${REPORT_DIR}/junit_report_${TIMESTAMP}.xml"
JSON_REPORT="${REPORT_DIR}/test_report_${TIMESTAMP}.json"

echo ""
echo "Running tests..."
echo ""

pytest tests/ \
    -v \
    --html=$HTML_REPORT \
    --self-contained-html \
    --junitxml=$JUNIT_REPORT \
    --json-report --json-report-file=$JSON_REPORT \
    --json-report-indent=2 \
    --tb=short \
    --durations=10 \
    -m "not slow"

echo ""
echo "=========================================="
echo "Test Reports Generated:"
echo "  - HTML Report: $HTML_REPORT"
echo "  - JUnit XML:   $JUNIT_REPORT"
echo "  - JSON Report: $JSON_REPORT"
echo "=========================================="

if command -v python &> /dev/null; then
    echo ""
    echo "Test Summary:"
    python -c "
import json
try:
    with open('$JSON_REPORT', 'r') as f:
        report = json.load(f)
    print(f\"  Total tests:  {report['summary']['total']}\")
    print(f\"  Passed:       {report['summary']['passed']}\")
    print(f\"  Failed:       {report['summary']['failed']}\")
    print(f\"  Skipped:      {report['summary']['skipped']}\")
    if report['summary']['failed'] > 0:
        print(f\"\n  ❌ Some tests failed!\")
        exit(1)
    else:
        print(f\"\n  ✅ All tests passed!\")
except Exception as e:
    print(f'Could not read JSON report: {e}')
"
fi

echo ""
echo "Done!"
