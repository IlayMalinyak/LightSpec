import torch
import torch.nn as nn
from collections import OrderedDict
from .Modules.mhsa_pro import MHA_rotary
from .Modules.cnn import ConvBlock
from nn.cnn import CNNEncoderDecoder, CNNEncoder
from nn.moco import MultimodalMoCo, LightCurveSpectraMoCo
from nn.simsiam import SimCLR, SimSiam, MultiModalSimSiam
from nn.astroconf import Astroconformer, AstroEncoderDecoder

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'SimCLR': SimCLR, 'SimSiam': SimSiam,
           'MultiModalSimSiam': MultiModalSimSiam, 'MultimodalMoCo': MultimodalMoCo,
             'LightCurveSpectraMoCo': LightCurveSpectraMoCo, 'AstroEncoderDecoder': AstroEncoderDecoder,
               'CNNEncoderDecoder': CNNEncoderDecoder,}

def init_model(model, model_args):
  if model_args.load_checkpoint:
      print("****Loading  checkpoint****")
      state_dict = torch.load(f'{model_args.checkpoint_path}', map_location=torch.device('cpu'))
      new_state_dict = OrderedDict()
      for key, value in state_dict.items():
          while key.startswith('module.'):
              key = key[7:]
          # key = key.replace('backbone.', '')
          new_state_dict[key] = value
      state_dict = new_state_dict
      missing, unexpected = model.load_state_dict(state_dict, strict=False)
      print("missing keys: ", missing)
      print("unexpected keys: ", unexpected)
  else:
      print("****deepnorm init****")
      deepnorm_init(model, model_args)
  return model

def deepnorm_init(model, args):

  def init_func(m):
    beta = getattr(args, 'beta', 1)
    if isinstance(m, MHA_rotary):  # adjust as necessary for your use case
      nn.init.xavier_normal_(m.query.weight, gain=1)
      nn.init.xavier_normal_(m.key.weight, gain=1)
      nn.init.xavier_normal_(m.value.weight, gain=beta)
      nn.init.xavier_normal_(m.output.weight, gain=beta)

      nn.init.zeros_(m.query.bias)
      nn.init.zeros_(m.key.bias)
      nn.init.zeros_(m.value.bias)
      nn.init.zeros_(m.output.bias)
      if getattr(m, 'ffn', None) is not None:
        nn.init.xavier_normal_(m.ffn.linear1.weight, gain=beta)
        nn.init.xavier_normal_(m.ffn.linear2.weight, gain=beta)
        nn.init.zeros_(m.ffn.linear1.bias)
        nn.init.zeros_(m.ffn.linear2.bias)

  model.apply(init_func)
