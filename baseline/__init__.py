__all__ = ["SAC", "CnnPolicy", "MlpPolicy", "MultiInputPolicy"]


def __getattr__(name: str):
    if name in __all__:
        from baseline.sac import SAC, CnnPolicy, MlpPolicy, MultiInputPolicy

        exports = {
            "SAC": SAC,
            "CnnPolicy": CnnPolicy,
            "MlpPolicy": MlpPolicy,
            "MultiInputPolicy": MultiInputPolicy,
        }
        return exports[name]
    raise AttributeError(f"module 'baseline' has no attribute {name!r}")
