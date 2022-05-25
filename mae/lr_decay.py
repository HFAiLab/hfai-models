import json


def param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.encoder.transformer) + 1
    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in ["encoder.cls_token", "encoder.pos_embed"]:
            use_decay = "no_decay"
            decay = 0.0
        else:
            use_decay = "decay"
            decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{use_decay}"

        if group_name not in param_group_names:
            scale = scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": scale,
                "weight_decay": decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": scale,
                "weight_decay": decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # groups = json.dumps(param_group_names, indent=2)
    # print(f"parameter groups: \n{groups}")
    # exit()

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    """
    if name in ["encoder.cls_token", "encoder.pos_embed"]:
        return 0
    elif name.startswith("encoder.patch_embed"):
        return 0
    elif name.startswith("encoder.transformer"):
        return int(name.split(".")[2]) + 1
    else:
        return num_layers
