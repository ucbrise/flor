from zipfile import ZipFile

from flor.log_scanner.scanner import Scanner

from . import settings

settings.init()

def test_is_subset():
    pass


def test_scan():

    with ZipFile(settings.logs_dir + 'log.json.zip', 'r') as zipObj:
        zipObj.extractall(settings.logs_dir)

    log_file_path = settings.logs_dir + 'log.json'

    scanner = Scanner(log_file_path)

    assert scanner.log_path == log_file_path

    scanner.scan_log()

    assert scanner.line_number == 7153