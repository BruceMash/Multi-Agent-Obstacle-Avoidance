__all__ = ["MASACNetworkConfig"]


def __getattr__(name):
    if name == "MASACNetworkConfig":
        from .config import MASACNetworkConfig

        return MASACNetworkConfig
    raise AttributeError(name)
