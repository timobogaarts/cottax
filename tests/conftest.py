'''
Every op in every test is applied under `fully_checked`: a proof in an op's `covers` that
passes what its rule refuses fails the test that applied it. The shipped features' answers
are registered once, for the suite.
'''

import pytest

import cottax.pytree.consumers, cottax.relational.consumers  # noqa: F401
from cottax.core.maths.plan.primitive import fully_checked


@pytest.fixture(autouse=True)
def _every_proof_checked():
    with fully_checked():
        yield
