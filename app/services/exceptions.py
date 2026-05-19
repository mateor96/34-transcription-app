class ProviderError(Exception):
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


class ProviderUnavailableError(ProviderError):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderModelError(ProviderError):
    pass
