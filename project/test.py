"""Testy jednostkowe dla project/main.py."""

from project.main import get_message, main


def test_get_message_returns_hello_world():
    assert get_message() == "Hello World"


def test_get_message_is_nonempty_string():
    msg = get_message()
    assert isinstance(msg, str)
    assert msg.strip() != ""


def test_main_prints_message(capsys):
    """main() powinno wypisać komunikat na stdout."""
    main()
    captured = capsys.readouterr()
    assert "Hello World" in captured.out


def test_main_prints_single_line(capsys):
    main()
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) == 1