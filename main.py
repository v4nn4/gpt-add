import fire

from gpt_add.train import train


class Trainer:
    def train(
        self,
        nb_samples_scoring: int = 100,
        max_iters: int = 50000,
        use_bigram: bool = False,
        model_size: str = "small",
        block_size: int = 12 * 10,
        batch_size: int = 32,
        eval_interval: int = 1000,
        learning_rate: float = 0.0005,
        eval_iters: int = 100,
        save_model: bool = False,
        operation: str = "add",
    ) -> None:
        train(
            nb_samples_scoring=nb_samples_scoring,
            max_iters=max_iters,
            model_size=model_size,
            use_bigram=use_bigram,
            block_size=block_size,
            batch_size=batch_size,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            eval_iters=eval_iters,
            save_model=save_model,
            operation=operation,
        )


if __name__ == "__main__":
    fire.Fire(Trainer)
