from classification.models import resnet50, mobilenetv2, densenet, efficientnet


class ModelsManeger:
    @staticmethod
    def get_model(model_name, args):
        models_dict = {
            "resnet50": resnet50.resnet50,
            "mobilenetv2": mobilenetv2.mobilenet_v2,
            "densenet": densenet.densenet3,
            "efficientnet": efficientnet.efficientnet
        }

        return models_dict[model_name](args)


