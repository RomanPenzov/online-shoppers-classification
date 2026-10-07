"""Сетевые подключения запрещены для всего offline-набора тестов."""

import socket

import pytest


@pytest.fixture(autouse=True)
def disable_network(monkeypatch):
    """Останови тест, если какой-либо код попробует подключиться к сети."""

    def blocked(*args, **kwargs):
        raise AssertionError("Unit-тесты не должны обращаться к сети")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
