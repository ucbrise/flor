from zipfile import ZipFile

import json

from flor.log_scanner.context import Ctx

from flor.log_scanner.scanner import Scanner

from . import settings

settings.init()

with ZipFile(settings.logs_dir + 'log.json.zip', 'r') as zipObj:
    zipObj.extractall(settings.logs_dir)

log_file_path = settings.logs_dir + 'log.json'


def test_is_subset():
    pass


def test_context():
    with open(log_file_path, 'r') as f:
        for idx, line in enumerate(f):
            log_record = json.loads(line.strip())
            if 'file_path' in log_record:
                ctx = Ctx()
                ctx.file_path = log_record['file_path']
                assert ctx.is_enabled(ctx)


def test_scan():
    scanner = Scanner(log_file_path)

    assert scanner.log_path == log_file_path

    scanner.scan_log()

    assert scanner.line_number == 7153
