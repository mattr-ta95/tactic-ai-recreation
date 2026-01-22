"""Custom PyTorch Lightning callbacks for TacticAI."""

from pytorch_lightning.callbacks import Callback


class WandbMetricsCallback(Callback):
    """Log additional metrics to W&B."""

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log extra metrics after validation."""
        try:
            import wandb
            if wandb.run is None:
                return

            # Log learning rate
            lr = trainer.optimizers[0].param_groups[0]['lr']
            wandb.log({'learning_rate': lr}, step=trainer.global_step)

            # Log random baseline comparison
            random_baseline = 1.0 / 16  # ~6.25% for ~16 players
            val_acc = trainer.callback_metrics.get('val/top1_acc', 0)
            if val_acc > 0:
                wandb.log({
                    'val/improvement_over_random': float(val_acc) / random_baseline,
                }, step=trainer.global_step)
        except ImportError:
            pass


class GradientNormCallback(Callback):
    """Log gradient norms for debugging."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log gradient norm before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        try:
            import wandb
            if wandb.run is None:
                return

            total_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            wandb.log({'gradient_norm': total_norm}, step=trainer.global_step)
        except ImportError:
            pass
