import pytest

_MARKER_NAME = "uses_virtual_device"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--with-virtual-device",
        dest="virtual_dev",
        action="store_true",
        help="Run tests suitable in a presence of test virtual device",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{_MARKER_NAME}: mark test to run only when virtual devices are available.",
    )

    import pyda_lsa._jpype_tools

    pyda_lsa._jpype_tools.set_lsa_server("next")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if list(
        item.iter_markers(name=_MARKER_NAME)
    ) and not item.config.getoption("virtual_dev", default=False):
        pytest.skip(
            "This relies on presence of a virtual device. For reproducibility, "
            "an LSA server must be connected for these tests",
        )
