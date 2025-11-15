from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingArguments:
    output_dir: str | None = field(
        default=None,
        metadata={
            'help': 'The output directory where the model predictions and checkpoints will be written.'
        },
    )
    eval_strategy: Literal['steps', 'epoch', 'no'] = field(
        default='no', metadata={'help': 'The evaluation strategy to use.'}
    )
    eval_steps: float = field(
        default=0.1,
        metadata={
            'help': 'Evaluate every X updates steps. If smaller than 1, will be interpreted as ratio of total training steps.'
        },
    )
    eval_on_start: bool = field(
        default=False,
        metadata={
            'help': 'Whether to run through the entire `evaluation` step at the very beginning of training as a sanity check.'
        },
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={'help': 'When performing evaluation and predictions, only returns the loss.'},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            'help': 'Number of updates steps to accumulate before performing a backward/update pass.'
        },
    )
    max_grad_norm: float = field(default=1.0, metadata={'help': 'Max gradient norm.'})
    num_train_epochs: int = field(
        default=3, metadata={'help': 'Total number of training epochs to perform.'}
    )
    max_steps: int = field(
        default=-1,
        metadata={
            'help': 'If > 0: set total number of training steps to perform. Override num_train_epochs.'
        },
    )
    run_name: str | None = field(
        default=None, metadata={'help': 'The name of the run. Used for logging.'}
    )
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = field(
        default='INFO', metadata={'help': 'The logging level.'}
    )
    logging_strategy: Literal['steps', 'epoch', 'no'] = field(
        default='steps', metadata={'help': 'The logging strategy to use.'}
    )
    logging_steps: float = field(
        default=500,
        metadata={
            'help': 'Log every X updates steps. If smaller than 1, will be interpreted as ratio of total training steps.'
        },
    )
    save_strategy: Literal['best', 'steps', 'epoch', 'no'] = field(
        default='steps', metadata={'help': 'The checkpoint save strategy to use.'}
    )
    save_steps: float = field(
        default=500,
        metadata={
            'help': 'Save checkpoint every X updates steps. If smaller than 1, will be interpreted as ratio of total training steps.'
        },
    )
    # save_total_limit: int = field(
    #     default=-1,
    #     metadata={
    #         "help": "If > 0, limit the total amount of checkpoints. Delete the older checkpoints in the output directory."
    #     },
    # )
    save_safetensors: bool = field(
        default=True,
        metadata={
            'help': 'Whether to save model in safetensors format instead of default torch.load and torch.save.'
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            'help': 'When checkpointing, whether to only save the model, or also the optimizer, scheduler && rng state.'
        },
    )
    seed: int = field(default=42, metadata={'help': 'Random seed.'})
    data_seed: int = field(default=42, metadata={'help': 'Random seed for data sampling.'})
    dataloader_drop_last: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the last incomplete batch in the dataloader.'},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            'help': 'Number of subprocesses to use for data loading (Pytorch only). 0 means that the data loader will use the main process.'
        },
    )
    disable_tqdm: bool = field(
        default=False, metadata={'help': 'Whether to disable tqdm progress bar.'}
    )
