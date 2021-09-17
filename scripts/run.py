import os
from dotmap import DotMap
from copy import deepcopy
from src.agents.purplebook import *
from src.agents.nlp import *
from src.utils.setup import process_config_from_json
from src.utils.utils import load_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args, gpu_device=0):
    config_json = load_json(args.config)
    config_json['gpu_device'] = gpu_device

    if args.iris_cluster:
        config_json['exp_base'] = config_json['exp_base'].replace('/mnt/fs5', '/iris/u')
        config_json['data_root'] = '/iris/u/wumike'
    else:
        config_json['data_root'] = '/data5/wumike'

    # config_json['dataset']['train']['add_cs106_assignments'] = args.cs106
    # config_json['dataset']['train']['cloze_tasks_factor'] = args.cloze_task
    # config_json['dataset']['train']['execution_tasks_factor'] = args.execution_task
    # config_json['dataset']['train']['roberta_rubric'] = not args.no_roberta
    # config_json['dataset']['train']['augment_by_names'] = not args.no_augment_by_names
    # config_json['dataset']['train']['augment_by_rubric'] = not args.no_augment_by_rubric
    config_json['dataset']['hold_out_split'] = not args.no_hold_out
    config_json['dataset']['hold_out_category'] = args.hold_out_category
    # config_json['dataset']['enforce_binary'] = not args.multiclass
    
    exp_name_suffix = build_exp_name_suffix(args)

    config = process_config_from_json(
        config_json,
        exp_name_suffix=exp_name_suffix,
    )
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    if config.continue_exp_dir is not None:
        agent.logger.info("Found existing model... Continuing training!")
        checkpoint_dir = os.path.join(config.continue_exp_dir, 'checkpoints')
        agent.load_checkpoint(config.continue_exp_name, checkpoint_dir=checkpoint_dir, 
                              load_model=True, load_optim=True, load_epoch=True)

    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass


def build_exp_name_suffix(args):
    exp_name_suffix = []
    # if args.cs106:
    #     exp_name_suffix.append('cs106')
    # if args.cloze_task > 0:
    #     exp_name_suffix.append(f'cloze_{args.cloze_task}')
    # if args.execution_task > 0:
    #     exp_name_suffix.append(f'exec_{args.execution_task}')
    if args.no_hold_out:
        exp_name_suffix.append('no_hold_out')
    else:  # exam | question
        exp_name_suffix.append(args.hold_out_category)
    # if args.no_roberta:
    #     exp_name_suffix.append('no_roberta')
    # if args.no_augment_by_names:
    #     exp_name_suffix.append('no_augment_by_names')
    # if args.no_augment_by_rubric:
    #     exp_name_suffix.append('no_augment_by_rubrics')
    # if args.multiclass:
    #     exp_name_suffix.append('multiclass')
    return '_'.join(exp_name_suffix)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    # parser.add_argument('--cs106', action='store_true', default=False)
    # parser.add_argument('--cloze-task', type=float, default=0)
    # parser.add_argument('--execution-task', type=float, default=0)
    parser.add_argument('--no-hold-out', action='store_true', default=False)
    parser.add_argument('--hold-out-category', type=str, choices=['exam', 'question'], default='exam')
    parser.add_argument('--iris-cluster', action='store_true', default=False)
    # parser.add_argument('--multiclass', action='store_true', default=False)
    # parser.add_argument('--no-roberta', action='store_true', default=False)
    # parser.add_argument('--no-augment-by-names', action='store_true', default=False)
    # parser.add_argument('--no-augment-by-rubric', action='store_true', default=False)
    parser.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()
    run(args, gpu_device=args.gpu_device)
