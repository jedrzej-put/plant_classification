import confuse
import inspect
from collections import abc

class ConfigLoader:
    """Load and validate config file using confuse library.
    """
    def __init__(self, config_path: str, template: dict, appname: str = 'config'):
        """Object creator.

        Parameters
        ----------
        config_path: str
            Path to config file.
        template: dict
            Validation template.
        appname: str, optional
            Application name. 
        """
        self.configuration = confuse.Configuration(appname, read=False)
        self.configuration.set_file(config_path)
        self.config = self.configuration.get(template)
        
    def get_config(self) -> dict:
        """
        Returns
        -------
        dict
            Validated config file.
        """
        return self.config
    
    def dump(self) -> str:
        """Useful for saving pretty (interpreted) version of yaml config file.
        Returns
        -------
        str
            Dumped config. 
        """
        return self.configuration.dump()

config_template = {
    'batch_size': confuse.Integer(), 
    'channels': confuse.TypeTemplate(list),
    'crop_scale': confuse.Number(),
    'epochs': confuse.Integer(),
    'early_stopping_patience': confuse.Integer(),
    'experiment': confuse.String(), 
    'experiment_name': confuse.String(), 
    'image_size': confuse.Integer(), 
    'learning_rate': confuse.Number(),
    'mlflow': confuse.TypeTemplate(bool, default=False), 
    'mlflow_params': confuse.TypeTemplate(dict),
    'reduce_lr_factor': confuse.Number(),
    'reduce_lr_patience': confuse.Integer(),
    'patch_size': confuse.Integer(),
    'seed': confuse.Integer(), 
    'shuffle': confuse.TypeTemplate(bool, default=False),
    'tensorboard': confuse.TypeTemplate(bool, default=False),
    'healthy_data_path': confuse.String(),
    'sick_data_path': confuse.String(),
}





# def function_wrapper(func, params):
#     func_wrapped = lambda *x: func(*x, **params)
#     func_wrapped.__name__ = func.__name__
#     return func_wrapped

# class ClassMapper(confuse.Template):
#     """Object mapper for classes and function.

#     """
#     def __init__(self, choices):
#         """Object creator.

#         Parameters
#         ----------
#         choices: dict
#             Dict with possible objects mapping.
#         """

#         self.choices = choices
#         self.template = confuse.MappingTemplate({
#             'name': confuse.String(),
#             'params': confuse.TypeTemplate(dict)
#         })

#     def convert(self, value, view):
#         """Validate dict with above template. Checks if 'name' is one of 
#         choices and return created function or class object.
#         """
#         try:
#             temp = self.template(view)
            
#             if value['name'] not in self.choices:
#                 self.fail(u'must be one of {0!r}, not {1!r}'.format(list(self.choices), value['name']), view)
                
#             if isinstance(self.choices, abc.Mapping):
#                 if inspect.isfunction(self.choices[temp.name]):
#                     return function_wrapper(self.choices[temp.name], temp.params)
#                 else:                    
#                     return self.choices[temp.name](**temp.params)
#             else:
#                 return value
            
#         except confuse.NotFoundError as e:
#             raise Exception(e)

#     def __repr__(self):
#         return 'ClassMapper({0!r})'.format(self.choices)

# class ListMapper(confuse.Template):
#     """Validate and map list of objects using ClassMapper.
#     """
#     def __init__(self, choices):        
#         self.choices = choices
#         self.template = ClassMapper(choices)

#     def convert(self, values, views):
#         out = []
#         for view in views:
#             temp = self.template(view)
#             out.append(temp)
#         return out

#     def __repr__(self):
#         return 'ListMapper({0!r})'.format(self.choices)


