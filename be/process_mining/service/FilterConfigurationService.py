from typing import Optional

from process_mining.model.FilterConfiguration import FilterConfiguration


class FilterConfigurationNotLoadedError(RuntimeError):
    """Raised when no FilterConfiguration has been set."""


class FilterConfigurationService:
    """Simple setter/getter helper for FilterConfiguration objects."""

    def __init__(self, config: Optional[FilterConfiguration] = None) -> None:
        self._config = config

    @property
    def is_loaded(self) -> bool:
        return self._config is not None

    def set_config(self, config: FilterConfiguration) -> None:
        if config is None:
            raise ValueError("config cannot be None")
        self._config = config

    def create(self, **kwargs) -> FilterConfiguration:
        self._config = FilterConfiguration(**kwargs)
        return self._config

    def update(self, **kwargs) -> FilterConfiguration:
        config = self._ensure_loaded()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def get_config(self) -> FilterConfiguration:
        return self._ensure_loaded()

    def clear(self) -> None:
        self._config = None

    def _ensure_loaded(self) -> FilterConfiguration:
        if self._config is None:
            raise FilterConfigurationNotLoadedError("FilterConfiguration has not been set")
        return self._config
