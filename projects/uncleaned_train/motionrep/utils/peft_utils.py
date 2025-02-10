import peft

from peft.utils.save_and_load import get_peft_model_state_dict
from peft import PeftModel


def save_peft_adaptor(model: peft.PeftModel, dir, save_base_model=False):
    # save the adaptor only
    model.save_pretrained(dir)

    if save_base_model:
        raise NotImplementedError


def load_peft_adaptor_and_merge(adaptor_path, base_model):
    model = PeftModel.from_pretrained(base_model, adaptor_path)
    model = model.merge_and_unload()

    return model


def _code_test_peft_load_save():
    import torch.nn as nn
    import torch
    import copy

    class MLP(nn.Module):
        def __init__(self, num_units_hidden=10):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(20, num_units_hidden),
                nn.ReLU(),
                nn.Linear(num_units_hidden, num_units_hidden),
                nn.ReLU(),
                nn.Linear(num_units_hidden, 2),
                nn.LogSoftmax(dim=-1),
            )

        def forward(self, X):
            return self.seq(X)

    module = MLP()
    print("=> Name of original model parameters:")
    for name, param in module.named_parameters():
        print(name, param.shape)
    module_copy = copy.deepcopy(module)
    config = peft.LoraConfig(
        r=8,
        target_modules=["seq.0", "seq.2"],
        modules_to_save=["seq.4"],
    )
    peft_model = peft.get_peft_model(module, config)

    peft_model.print_trainable_parameters()

    print("\n=> Name of PeftModel's parameters:")
    for name, param in peft_model.named_parameters():
        print(name, param.shape)

    save_path = "./tmp"

    save_peft_adaptor(peft_model, save_path)

    loaded_merged_model = load_peft_adaptor_and_merge(save_path, module_copy)

    print("\n=> Name of Loaded and Merged model's parameters:")
    for name, param in loaded_merged_model.named_parameters():
        print(name, param.shape)


if __name__ == "__main__":
    _code_test_peft_load_save()
