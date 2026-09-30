import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def test_external_network_is_blocked():
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("93.184.215.14", 443), timeout=1)
