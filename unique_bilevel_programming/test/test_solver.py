import pytest


inst_1 = 1
weigths_1 = 1
outputs_1 = 1


@pytest.mark.parametrize(
    'instance,weights,outputs',
    [
        [inst_1, weigths_1, outputs_1],
    ]
)
def test_simple_instance_1(instance, weights, outputs):
    assert instance == outputs + 1
