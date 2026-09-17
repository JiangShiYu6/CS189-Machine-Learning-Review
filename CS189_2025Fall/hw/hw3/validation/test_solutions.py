"""Check notebook definitions without network access or training (pytest)."""
from pathlib import Path
import doctest
import json

import numpy as np
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "hw3.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
NS = {"__name__": __name__}
exec("from __future__ import annotations\nimport numpy as np\nfrom dataclasses import dataclass", NS)
for index in (6, 12, 13, 16):
    exec(compile("from __future__ import annotations\n" + "".join(NB["cells"][index]["source"]),
                 f"hw3.ipynb:cell{index}", "exec"), NS)
BearTensor = NS["BearTensor"]


@pytest.mark.parametrize("question", ["q1", "q2", "q3"])
def test_embedded_public_cases(question):
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner()
    for suite in NB["metadata"]["otter"]["tests"][question]["suites"]:
        for i, case in enumerate(suite["cases"]):
            test = parser.get_doctest(case["code"], NS.copy(), f"{question}-{i}", str(NOTEBOOK), 0)
            runner.run(test)
    assert runner.failures == 0


def check_gradient(fn, arrays):
    nodes = [BearTensor(f"x{i}", a.copy()) for i, a in enumerate(arrays)]
    fn(*nodes).sum().backward()
    for node, original in zip(nodes, arrays):
        numeric = np.zeros_like(original, dtype=float)
        for index in np.ndindex(original.shape):
            h = 1e-6
            node.value[index] += h
            plus = fn(*nodes).value.sum()
            node.value[index] -= 2 * h
            minus = fn(*nodes).value.sum()
            node.value[index] = original[index]
            numeric[index] = (plus - minus) / (2 * h)
        np.testing.assert_allclose(node.adjoint, numeric, atol=2e-6, rtol=2e-5)
        assert node.adjoint.shape == original.shape


@pytest.mark.parametrize("fn", [
    lambda a, b: ((a * b + a) - b) ** 2,
    lambda a, b: (a * b).sigmoid().mean(),
    lambda a, b: (a - b).relu().sum(),
    lambda a, b: (a ** -1) * (b ** 0.5),
])
def test_composite_gradients(fn):
    check_gradient(fn, [np.array([0.4, 1.2, 2.5]), np.array([1.0, 0.3, 1.8])])


@pytest.mark.parametrize("shapes", [
    ((3, 2), (2, 4)), ((3, 2), (2,)), ((3,), (1, 2)),
    ((), (1, 2)), ((2, 3, 4), (4, 2)), ((1, 3, 4), (2, 4, 2)),
])
def test_matmul_gradients(shapes):
    rng = np.random.default_rng(7)
    check_gradient(lambda a, b: (a @ b).sigmoid().mean(),
                   [np.asarray(rng.normal(size=shape)) for shape in shapes])


def test_dot_gradient():
    check_gradient(lambda a, b: a.dot(b) ** 2,
                   [np.array([1., 2., -1.]), np.array([0.3, -0.5, 0.8])])


def test_shared_graph_and_repeated_backward():
    x = BearTensor("x", np.array([2., 3.]))
    shared = x * x
    loss = (shared * shared + shared).sum()
    for _ in range(2):
        loss.backward()
        np.testing.assert_allclose(x.adjoint, 4 * x.value ** 3 + 2 * x.value)
    loss.reset_children()
    for node in NS["topological_sort"](loss):
        np.testing.assert_array_equal(node.adjoint, np.zeros_like(node.value))


def test_deep_graph_without_recursion():
    x = BearTensor("x", np.array([1.]))
    one = BearTensor("one", np.array([1.]))
    out = x
    for _ in range(2500):
        out = out + one
        out.name = "chain"
    out.backward()
    np.testing.assert_array_equal(x.adjoint, [1.])
    np.testing.assert_array_equal(one.adjoint, [2500.])


def test_extreme_sigmoid_and_zero_power():
    x = BearTensor("x", np.array([-1000., 0., 1000.]))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s = x.sigmoid()
        s.backward()
        np.testing.assert_allclose(s.value, [0., 0.5, 1.])
        np.testing.assert_allclose(x.adjoint, [0., 0.25, 0.])
        (x ** 0).backward()
        np.testing.assert_array_equal(x.adjoint, [0., 0., 0.])


def test_adam_written_example():
    x = BearTensor("x", np.array([10.]))
    adam = NS["Adam"]([x], lr=0.1, beta2=0.99)
    for _ in range(2):
        (x ** 2).sum().backward()
        adam.step()
    np.testing.assert_allclose(x.value, [9.800025185], atol=1e-9)
    assert adam.t == 2


def test_shape_errors():
    a = BearTensor("a", np.ones(2))
    b = BearTensor("b", np.ones((2, 1)))
    for operation in (lambda: a + b, lambda: a - b, lambda: a * b, lambda: a.dot(b)):
        with pytest.raises(ValueError):
            operation()
