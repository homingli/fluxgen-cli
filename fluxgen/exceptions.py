class FluxgenError(Exception):
    """Base exception for all fluxgen errors."""
    pass

class ModelLoadError(FluxgenError):
    """Raised when a model fails to load."""
    pass

class InvalidImageError(FluxgenError):
    """Raised when an input image is invalid or corrupted."""
    pass

class PathTraversalError(FluxgenError):
    """Raised when an output path attempts to write outside the intended directory."""
    pass

class InvalidConfigurationError(FluxgenError):
    """Raised when there is an invalid configuration or argument."""
    pass
