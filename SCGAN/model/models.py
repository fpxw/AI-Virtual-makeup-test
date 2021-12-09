from .SCGAN import SCGAN
def create_model(opt,dataset):
    model = SCGAN(dataset)
    # print('model-ok')
    # print(opt)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
