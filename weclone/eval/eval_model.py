from llamafactory.eval.evaluator import Evaluator
from llamafactory.hparams import get_eval_args

from weclone.train.security import model_runtime, parsed_cli_arguments
from weclone.utils.secure_storage import is_encrypted_mode


def main():
    if not is_encrypted_mode():
        Evaluator().eval()
        return
    with model_runtime(parsed_cli_arguments(get_eval_args), evaluation=True) as runtime:
        Evaluator(runtime).eval()


if __name__ == "__main__":
    main()
