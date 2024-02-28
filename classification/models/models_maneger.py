from classification.models import resnet50, mobilenetv2, densenet, efficientnet
import torch


class ModelsManeger:
    @staticmethod
    def get_model(model_name, args):
        models_dict = {
            "resnet50": resnet50.resnet50,
            "mobilenetv2": mobilenetv2.mobilenet_v2,
            "densenet121": densenet.densenet121,
            "densenet169": densenet.densenet169,
            "densenet201": densenet.densenet201,
            "densenet161": densenet.densenet161,
            "efficientnet": efficientnet.efficientnet
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return models_dict[model_name](args).to(device)


