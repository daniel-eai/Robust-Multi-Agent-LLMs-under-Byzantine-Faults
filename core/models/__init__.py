
from .base_model import BaseModel, NetworkClient
from .api_model import APIModel
from .local_model import LocalModel

__all__ = [
    'BaseModel',
    'NetworkClient', 
    'APIModel',
    'LocalModel',
    'create_model'
]

def create_model(model_type: str = 'api', **kwargs):

    if model_type.lower() == 'api':
        return APIModel(**kwargs)
    elif model_type.lower() == 'local':
        return LocalModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'api', 'local'")

__version__ = "2.0.0" 