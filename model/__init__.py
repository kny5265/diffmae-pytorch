import importlib

def find_model_using_name(model_name):
    model_filename = "model." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def create_model(args):
    model = find_model_using_name(args.model)
    instance = model(args)
    print("model [{}] was created".format(type(instance).__name__))
    return instance