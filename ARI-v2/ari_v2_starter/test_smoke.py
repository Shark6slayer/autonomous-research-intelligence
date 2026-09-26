import tempfile
from pathlib import Path
from ari.main import make_report, search_queries

def test_basics():
    assert search_queries('Can transformers improve RF signal classification?')
    report = make_report('Test question', [{'title':'Example Paper','year':2025,'url':'https://example.org/paper','abstract':'An abstract.'}])
    assert 'Example Paper' in report and 'https://example.org/paper' in report
    assert 'full-text verification' in report

if __name__ == '__main__':
    test_basics()
    print('Smoke tests passed')
