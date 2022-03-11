import os
from src.agents.feedback import *
from src.agents.nlp import *
from src.utils.setup import process_config_from_json
from src.utils.utils import load_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args, gpu_device=0):
    config_json = load_json(args.config)
    config_json['gpu_device'] = gpu_device
    config = process_config_from_json(config_json)
    AgentClass = globals()[config.agent]
    print("CONFIGURING AGENT")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()
    run(args, gpu_device=args.gpu_device)
