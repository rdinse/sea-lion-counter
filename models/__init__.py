from models.basic_model import BasicModel
from models.countception_model import CountCeptionModel
from models.inception_model import InceptionModel
from models.contextual_inception_model import ContextualInceptionModel

__all__ = [
    "CountCeptionModel",
    "InceptionModel",
    "ContextualInceptionModel",
]

def make_model(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']](config)
    else:
        raise Exception('The model name %s does not exist' % config['model_name'])

def get_model_class(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']]
    else:
        raise Exception('The model name %s does not exist' % config['model_name'])
