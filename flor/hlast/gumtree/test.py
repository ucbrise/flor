#!/usr/bin/env python
# type: ignore
from itertools import zip_longest

from . import GumTree, Mapping
from .tree import Node, adapter


N = Node
gt = GumTree(adapter)


def test():
    t1, t2 = example()

    result = gt.topdown(t1, t2)

    expected = Mapping(adapter)
    expected.put_tree(t1[0][2][1], t2[0][2][1])
    expected.put_tree(t1[0][2][3], t2[0][2][3])
    expected.put_tree(t1[0][2][4][0][0], t2[0][2][4][0][0])
    expected.put_tree(t1[0][2][4][0][1], t2[0][2][4][0][2][1])
    assert match(result, expected), "topdown"

    result = Mapping(adapter, expected)
    gt.bottomup(t1, t2, result)

    expected.put(t1[0][2][4][0], t2[0][2][4][0])
    expected.put(t1[0][2][4], t2[0][2][4])
    expected.put(t1[0][2], t2[0][2])
    expected.put(t1[0][2][0], t2[0][2][0])
    expected.put(t1[0][2][2], t2[0][2][2])
    expected.put(t1[0], t2[0])
    expected.put(t1[0][0], t2[0][0])
    expected.put(t1[0][1], t2[0][1])
    expected.put(t1, t2)
    assert match(result, expected), "bottomup"

    assert match(gt.mapping(t1, t2), expected)
    print("Passed Example!")


def example():
    source = N(
        "CompilationUnit",
        "",
        [
            N(
                "TypeDeclaration",
                "",
                [
                    N("Modifier", "public"),
                    N("SimpleName", "Test"),
                    N(
                        "MethodDeclaration",
                        "",
                        [
                            N("Modifier", "private"),
                            N(
                                "SimpleType",
                                "String",
                                [
                                    N("SimpleName", "String"),
                                ],
                            ),
                            N("SimpleName", "foo"),
                            N(
                                "SingleVariableDeclaration",
                                "",
                                [
                                    N("PrimitiveType", "int"),
                                    N("SimpleName", "i"),
                                ],
                            ),
                            N(
                                "Block",
                                "",
                                [
                                    N(
                                        "IfStatement",
                                        "",
                                        [
                                            N(
                                                "InfixExpression",
                                                "==",
                                                [
                                                    N("SimpleName", "i"),
                                                    N("NumberLiteral", "0"),
                                                ],
                                            ),
                                            N(
                                                "ReturnStatement",
                                                "",
                                                [
                                                    N("StringLiteral", "Foo!"),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
    destination = N(
        "CompilationUnit",
        "",
        [
            N(
                "TypeDeclaration",
                "",
                [
                    N("Modifier", "public"),
                    N("SimpleName", "Test"),
                    N(
                        "MethodDeclaration",
                        "",
                        [
                            N("Modifier", "private"),
                            N(
                                "SimpleType",
                                "String",
                                [
                                    N("SimpleName", "String"),
                                ],
                            ),
                            N("SimpleName", "foo"),
                            N(
                                "SingleVariableDeclaration",
                                "",
                                [
                                    N("PrimitiveType", "int"),
                                    N("SimpleName", "i"),
                                ],
                            ),
                            N(
                                "Block",
                                "",
                                [
                                    N(
                                        "IfStatement",
                                        "",
                                        [
                                            N(
                                                "InfixExpression",
                                                "==",
                                                [
                                                    N("SimpleName", "i"),
                                                    N("NumberLiteral", "0"),
                                                ],
                                            ),
                                            N(
                                                "ReturnStatement",
                                                "",
                                                [
                                                    N("StringLiteral", "Bar"),
                                                ],
                                            ),
                                            N(
                                                "IfStatement",
                                                "",
                                                [
                                                    N(
                                                        "InfixExpression",
                                                        "==",
                                                        [
                                                            N("SimpleName", "i"),
                                                            N(
                                                                "PrefixExpression",
                                                                "-",
                                                                [
                                                                    N(
                                                                        "NumberLiteral",
                                                                        "1",
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    N(
                                                        "ReturnStatement",
                                                        "",
                                                        [
                                                            N("StringLiteral", "Foo!"),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
    return source, destination


def match(left: Mapping, right: Mapping):
    return all(
        id(el) == id(rl) and id(er) == id(rr)
        for (el, er), (rl, rr) in zip_longest(left.items(), right.items())
    )


if __name__ == "__main__":
    test()
