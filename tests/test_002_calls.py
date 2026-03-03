# pylint: disable=import-outside-toplevel,missing-module-docstring,missing-function-docstring


def test_package_imports():
    import acherus

    assert isinstance(acherus.__version__, str)
    assert len(acherus.__version__) > 0


def test_cli_module_imports():
    import acherus.__main__
    import acherus.monitoring
    import acherus.plotting

    assert callable(acherus.__main__.main)
    assert callable(acherus.monitoring.main)
    assert callable(acherus.plotting.main)
