import ast
import astor

from flor.state_machine_generator.handler import Handler

valid_highlights = [
    "GET('a', 0.001)",
    "GET('test_size', 0.2)",
    "GET('test_acc', score)"
]

valid_highlight_names = [
    'a', 'test_size', 'test_acc'
]


def test_is_highlight():
    invalid_highlights = [
        "GET(name, 0.001)",
        "FlorGET(another_name, 'hello world')"
    ]

    for source in valid_highlights:
        tree = ast.parse(source, mode='exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                h = Handler(node)
                assert h.is_highlight()
                assert not h.is_flog_write()

    for source in invalid_highlights:
        tree = ast.parse(source, mode='exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                h = Handler(node)
                assert not h.is_highlight()
                assert not h.is_flog_write()


def test_fetch_highlight_name():
    for source, name in zip(valid_highlights, valid_highlight_names):
        tree = ast.parse(source, mode='exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert Handler(node).fetch_highlight_name() == name


def test_fetch_lsn():
    source = "Flog.flagged() and flog.write({'locals': [{'iris': flog.serialize(iris)}], 'lineage': 'iris = datasets.load_iris()', 'lsn': 2})"
    tree = ast.parse(source, mode='exec')
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            h = Handler(node)
            assert h.is_flog_write()
            assert h.fetch_lsn() == 2
