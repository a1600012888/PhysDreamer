from omegaconf import OmegaConf


def load_config_with_merge(config_path: str):
    cfg = OmegaConf.load(config_path)

    path_ = cfg.get("_base", None)

    if path_ is not None:
        print(f"Merging base config from {path_}")
        cfg = OmegaConf.merge(load_config_with_merge(path_), cfg)
    else:
        return cfg
    return cfg


def merge_without_none(base_cfg, override_cfg):
    for key, value in override_cfg.items():
        if value is not None:
            base_cfg[key] = value
        elif not (key in base_cfg):
            base_cfg[key] = None
    return base_cfg


def create_config(config_path, args, cli_args: list = []):
    """
    Args:
        config_path: path to config file
        args: argparse object with known variables
        cli_args: list of cli args in the format of
            ["lr=0.1", "model.name=alexnet"]
    """
    # recursively merge base config
    cfg = load_config_with_merge(config_path)

    # parse cli args, and merge them into cfg
    cli_conf = OmegaConf.from_cli(cli_args)
    arg_cfg = OmegaConf.create(vars(args))

    # drop None in arg_cfg

    arg_cfg = OmegaConf.merge(arg_cfg, cli_conf)

    # cfg = OmegaConf.merge(cfg, arg_cfg, cli_conf)
    cfg = merge_without_none(cfg, arg_cfg)

    return cfg
