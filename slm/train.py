import logging
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from data import MidiDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from tokenizer import Tokenizer
import einops
from fractions import Fraction
from masking import (
    mlm_mask,
    random_add_masking_mml,
    random_add_masking_variable_superposition,
    attribute_masking,
    event_masking,
    mixed_superposition,
    mixed_superposition_2,
    ratio_superposition,
    simple_superposition,
)
from model import SuperposedLanguageModel
from mdlm import LogLinearNoise


class TrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model_config,
        learning_rate,
        learning_rate_gamma,
        masking_scheme,
        use_weight_decay=False,
        attribute_masking_prob=0.0,
        event_masking_prob=0.0,
        warmup_steps=0,
        lr_steps_per_epoch=2836,
        collapse_inactive_events=False,
        loss = "cross_entropy"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SuperposedLanguageModel(**model_config)
        self.learning_rate = learning_rate
        self.learning_rate_gamma = learning_rate_gamma
        self.lr_steps_per_epoch = lr_steps_per_epoch
        self.use_mlm = self.model.use_mlm
        self.masking_scheme = masking_scheme
        self.tokenizer = self.model.tokenizer
        self.syntax_mask = torch.Tensor(self.tokenizer.get_syntax_mask())
        self.use_weight_decay = use_weight_decay
        self.attribute_masking_prob = attribute_masking_prob
        self.event_masking_prob = event_masking_prob
        self.warmup_steps = warmup_steps
        self.loss = loss

        # assert that masking_scheme is compatible with use_mlm
        if self.use_mlm:
            assert (
                self.masking_scheme == "mlm"
                or self.masking_scheme == "mdlm"
                or self.masking_scheme == "mlm_mixed_masking_2"
                or self.masking_scheme == "mlm_ratio_superposition_mixed_h_mixed_s"
            )
        else:
            assert (
                self.masking_scheme == "variable_superposition"
                or self.masking_scheme == "mml"
                or self.masking_scheme == "variable_superposition_x**1/2"
                or self.masking_scheme == "variable_superposition_x**1/4"
                or self.masking_scheme == "mixed_superposition"
                or self.masking_scheme == "mixed_superposition_2"
                or self.masking_scheme == "mixed_superposition_2_all_full"
                or self.masking_scheme == "simulated_mlm"
                or self.masking_scheme == "mixed_superposition_2_all_sparse"
                or self.masking_scheme == "ratio_superposition_mixed_h_mixed_s"
                or self.masking_scheme == "ratio_superposition_mixed_h_full_s"
                or self.masking_scheme == "ratio_superposition_mixed_h_mixed_s_w_shared_rate"
                or self.masking_scheme == "ratio_superposition_mixed_h_mixed_s_w_shared_rate_&_autoregression"
                or self.masking_scheme == "simple_superposition_x**1/2"
                or self.masking_scheme == "simple_superposition_x**1/4"
                or self.masking_scheme == "simple_superposition"
            )
        pass

        self.collapse_inactive_events = collapse_inactive_events

    def get_model_dtype(self):
        return self.model.embedding_layer.weight.dtype

    def get_model_device(self):
        return next(self.model.parameters()).device

    def step(self, batch, batch_idx):
        """
        Perform a single forward pass and calculate metrics
        Input:
            batch: dictionary containing batch data (token_ids (batch_size, events, attributes), n_loops_in_parent_song)
            batch_idx: index of batch
        """
        if self.masking_scheme == "mdlm":
            return self.step_mdlm(batch, batch_idx)
        token_ids = batch["token_ids"]
        target_token_ids = batch["token_ids"].to(self.get_model_device())
        # one hot encode token_ids with model dtype
        x = torch.nn.functional.one_hot(
            token_ids, num_classes=len(self.tokenizer.vocab)
        ).to(self.get_model_dtype())
        # apply masking scheme
        if self.use_mlm:
            if self.masking_scheme == "mlm_mixed_masking_2":
                x_input = mixed_superposition_2(x, mlm=True)
            elif self.masking_scheme == "mlm_ratio_superposition_mixed_h_mixed_s":
                x_masked = ratio_superposition(x, syntax_mask=self.syntax_mask)
                x_masked = x_masked * self.syntax_mask[None, None, ...].to(self.get_model_dtype()).to(self.get_model_device())
                x_prior = x_masked / x_masked.sum(dim=-1, keepdim=True)
                mlm_mask = (x_masked.sum(dim=-1, keepdim=True) > 1).to(self.get_model_dtype())
                x_masked = torch.cat([x_masked * (1 - mlm_mask), mlm_mask], dim=-1)
            else:
                x_ta = einops.rearrange(x, "b t a v -> b (t a) v")
                x_masked_ta = mlm_mask(x_ta, mask_first=False)
                x_input = einops.rearrange(
                    x_masked_ta,
                    "b (t a) vm -> b t a vm",
                    a=len(self.tokenizer.note_attribute_order),
                )
        else:
            if self.masking_scheme == "mixed_superposition":
                x_masked = mixed_superposition(x)
            elif self.masking_scheme == "simple_superposition":
                x_masked = simple_superposition(x, syntax_mask=self.syntax_mask, superpositions=["full", "sparse"])
            elif self.masking_scheme == "simple_superposition_x**1/2":
                x_masked = simple_superposition(x, syntax_mask=self.syntax_mask, superpositions=["full", "sparse"], schedule_fn = lambda x: x**(1/2))
            elif self.masking_scheme == "simple_superposition_x**1/4":
                x_masked = simple_superposition(x, syntax_mask=self.syntax_mask, superpositions=["full", "sparse"], schedule_fn = lambda x: x**(1/4))
            elif self.masking_scheme == "ratio_superposition_mixed_h_mixed_s":
                x_masked = ratio_superposition(x, syntax_mask=self.syntax_mask)
            elif self.masking_scheme == "ratio_superposition_mixed_h_mixed_s_w_shared_rate":
                x_masked = ratio_superposition(x, syntax_mask=self.syntax_mask, superpositions = ["full","full","sparse","shared_rate"])
            elif self.masking_scheme == "ratio_superposition_mixed_h_mixed_s_w_shared_rate_&_autoregression":
                x_masked = ratio_superposition(x, syntax_mask=self.syntax_mask, superpositions = ["full","full","sparse","shared_rate"], simulate_autoregression=True)
            elif self.masking_scheme == "ratio_superposition_mixed_h_full_s":
                x_masked = ratio_superposition(x, superpositions=["full"], syntax_mask=self.syntax_mask)
            elif self.masking_scheme == "mixed_superposition_2_all_sparse":
                x_masked = mixed_superposition_2(x, mlm=False, second_mask_types=["variable_superposition"])
            elif self.masking_scheme == "mixed_superposition_2":
                x_masked = mixed_superposition_2(x)
            elif self.masking_scheme == "mixed_superposition_2_all_full":
                x_masked = mixed_superposition_2(x, mlm=False, second_mask_types=["full"])
            elif self.masking_scheme == "simulated_mlm":
                x_masked = mixed_superposition_2(x, mlm=False, hierarchy_mask_types=["event_attribute"], second_mask_types=["full"])
            else:
                x_ta = einops.rearrange(x, "b t a v -> b (t a) v")
                if self.masking_scheme == "mml":
                    x_masked_ta = random_add_masking_mml(x_ta)
                elif self.masking_scheme == "variable_superposition":
                    x_masked_ta = random_add_masking_variable_superposition(x_ta)
                elif self.masking_scheme == "variable_superposition_x**1/2":
                    x_masked_ta = random_add_masking_variable_superposition(
                        x_ta, lambda x: x**0.5
                    )
                elif self.masking_scheme == "variable_superposition_x**1/4":
                    x_masked_ta = random_add_masking_variable_superposition(
                        x_ta, lambda x: x**0.25
                    )
                x_masked = einops.rearrange(
                    x_masked_ta,
                    "b (t a) v -> b t a v",
                    a=len(self.tokenizer.note_attribute_order),
                )
            if self.attribute_masking_prob > 0:
                x_masked = attribute_masking(x_masked, self.attribute_masking_prob)
            if self.event_masking_prob > 0:
                x_masked = event_masking(x_masked, self.event_masking_prob)
            x_masked = x_masked * self.syntax_mask[None, None, ...].to(self.get_model_dtype()).to(self.get_model_device())
           
            # renormalize
            x_prior = x_masked / x_masked.sum(dim=-1, keepdim=True)
         
        if self.use_mlm:
            logits = self.model(x_masked)
        else:
            logits = self.model(x_prior)

        # get metrics
        metrics = self.compute_metrics(
            target_token_ids, logits , prior=None if self.use_mlm else x_prior
        )

        if self.use_mlm:
            # apply constraint to logits
            with torch.no_grad():
                filtered_logits = logits.clone().detach()
                zero = torch.zeros_like(
                    x_prior, dtype=x_prior.dtype, device=x_prior.device
                )
                filtered_logits[x_prior.isclose(zero, rtol=1e-5)] = torch.finfo(
                    logits.dtype
                ).min
                filtered_logits_metrics = self.compute_metrics(
                    target_token_ids, filtered_logits
                )
                # add filtered logits metrics to metrics
                for key in filtered_logits_metrics:
                    metrics[f"filtered_logits/{key}"] = filtered_logits_metrics[key]
        return metrics

    def stage_step(self, batch, batch_idx, stage):
        metrics = self.step(batch, batch_idx)
        # log metrics
        for metric in metrics:
            self.log(
                f"{stage}/{metric}",
                metrics[metric],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        # calculate loss
        loss = metrics[self.loss]
        # plot loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        return self.stage_step(batch, batch_idx, "trn")

    def validation_step(self, batch, batch_idx):
        self.eval()
        return self.stage_step(batch, batch_idx, "val")

    def compute_metrics(self, target_token_ids, logits, prior=None):
        """
        Compute metrics for a given batch
        Input:
            target_token_ids: target token ids (batch_size, events, attributes)
            logits: model logits (batch_size, events, attributes, vocab_size)
            n_loops_in_parent_song: number of loops in parent song (batch_size)
        Returns:
            Dictionary containing metrics:
            - Cross entropy (global and per attribute)
            - Entropy (global and per attribute)
            - Probability (global and per attribute)
            - Accuracy @1, @2, @4 (global and per attribute)
        """
        batch_size, n_events, n_attributes = target_token_ids.shape

        # Calculate cross entropy
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_token_ids.reshape(-1),
            reduction="none",
        )
      
        ce = ce.reshape(batch_size, n_events * n_attributes)

        metrics = {}
        metrics["cross_entropy"] = ce.mean()

        if prior is not None:
            target_onehot = F.one_hot(target_token_ids, num_classes=logits.shape[-1])
            prior_prob = (prior * target_onehot).sum(-1)
            prior_log_prob = torch.log(prior_prob)
            # reshape to (batch_size, events attributes)
            prior_log_prob = einops.rearrange(prior_log_prob, "b e a -> b (e a)", e=n_events, a=n_attributes)
            metrics["prior_cross_entropy"] = - prior_log_prob.mean()
            metrics["cross_entropy_normed_pos"] = (ce / (1 - prior_log_prob)).mean()
            metrics["cross_entropy_normed_set"] = (ce.mean(-1) / (1 - prior_log_prob).mean(-1)).mean()

        with torch.no_grad():
            # Calculate probabilities and entropy
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * (log_probs + 1e-8)).sum(
                dim=-1
            )  # (batch_size, events, attributes)

            # Get probability of correct tokens
            target_probs = torch.gather(
                probs.reshape(-1, logits.shape[-1]),
                dim=-1,
                index=target_token_ids.reshape(-1).unsqueeze(-1),
            ).reshape(batch_size, n_events, n_attributes)

            # Initialize metrics dictionary
            metrics.update(
                {
                    "entropy": entropy.mean(),
                    "probability": target_probs.mean(),
                }
            )

            # Calculate accuracies for k=[1,2,4]
            topk_values = [1, 2, 4]
            for k in topk_values:
                decoder_output_probs_sort = torch.argsort(
                    probs, dim=-1, descending=True
                )  # (batch_size, events, attributes, vocab_size)
                accuracy = (
                    (
                        target_token_ids.unsqueeze(-1)
                        == decoder_output_probs_sort[..., :k]
                    )
                    # (batch_size, events, attributes, k)
                    .any(dim=-1)  # (batch_size, events, attributes)
                    .float()
                )

                metrics[f"accuracy@{k}"] = accuracy.mean()

                # Per-attribute accuracy metrics
                for i, attr in enumerate(self.tokenizer.note_attribute_order):
                    metrics[f"accuracy@{k}/{attr}"] = accuracy[:, :, i].mean()

            # Per-attribute metrics
            for attr_idx, attribute in enumerate(self.tokenizer.note_attribute_order):
                # Per-attribute entropy
                metrics[f"entropy/{attribute}"] = entropy[:, :, attr_idx].mean()

                # Per-attribute probability
                metrics[f"probability/{attribute}"] = target_probs[
                    :, :, attr_idx
                ].mean()

        return metrics

    def configure_optimizers(self):
        if self.use_weight_decay:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            # Post-warmup phase: start decay from step 1 after warmup
            decay_steps = current_step - self.warmup_steps
            return self.learning_rate_gamma ** (decay_steps / self.lr_steps_per_epoch)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # Update at every step
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def generate(
        self,
        x=None,
        sampling_steps=None,
        temperature=None,
        top_p=None,
        top_k=None,
        order="random",
        attribute_temperature=None,
        tokens_per_step=None,
    ):
        return self.model.generate(
            x,
            sampling_steps,
            temperature,
            top_p,
            top_k,
            order,
            attribute_temperature,
            tokens_per_step,
        )

    def convert_mlm_to_slm(self):
        """
        Converts the model from MLM to SLM architecture. This involves:
        1. Removing the mask token dimension from embedding layer
        2. Updating the model configuration
        3. Preserving compatible weights
        """
        if not self.use_mlm:
            print("Model is already an SLM, no conversion needed")
            return

        print("Converting MLM to SLM...")

        # Store original embedding weights without mask token dimension
        orig_embedding = self.model.embedding_layer.weight[:, :-1].clone()
        orig_unembedding = self.model.unembedding_layer.weight.clone()

        # Store transformer weights
        transformer_state = self.model.main_block.state_dict()

        # Update model config
        self.model.use_mlm = False

        # Create new embedding layer without mask token dimension
        self.model.embedding_layer = torch.nn.Linear(
            len(self.tokenizer.vocab),
            self.model.embedding_layer.weight.size(0),
            bias=False,
        )

        # Initialize with stored weights
        with torch.no_grad():
            self.model.embedding_layer.weight.copy_(orig_embedding)
            self.model.unembedding_layer.weight.copy_(orig_unembedding)

        # Restore transformer weights
        self.model.main_block.load_state_dict(transformer_state)

        # Update training wrapper attributes
        self.use_mlm = False
        self.masking_scheme = "variable_superposition"

        print("Successfully converted MLM to SLM architecture")

import argparse
if __name__ == "__main__":

    # optionally take in the name of a checkpoint and devices to use
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    # if checkpoint ends with .ckpt keep it, otherwise make it ./checkpoints/{checkpoint}/last.ckpt
    if checkpoint is not None and not checkpoint.endswith(".ckpt"):
        checkpoint = f"./checkpoints/{checkpoint}/last.ckpt"
    devices = args.devices
    if len(devices) == 0:
        devices = [0,1]

    SEED = 0

    torch.manual_seed(SEED)

    DATASET = "mmd_loops"
    BATCH_SIZE = 60

    N_BARS = 4 if DATASET == "harmonic" else 4

    if checkpoint is None:

        tag_list = open(f"./data/{DATASET}/tags.txt").read().splitlines()

        tokenizer_config = {
            "ticks_per_beat": 24
            if (DATASET == "mmd_loops" or DATASET == "harmonic")
            else 48,
            "time_hierarchy": "tick",
            "pitch_range": [0, 128],
            "max_beats": 4 * N_BARS,
            "max_notes": 75 * N_BARS if DATASET == "mmd_loops" else 20 * N_BARS,
            "min_tempo": 40,
            "max_tempo": 300,
            "n_tempo_bins": 32,
            "n_velocity_bins": 32,
            "time_signatures": None,
            "tags": tag_list,
            "shuffle_notes": True,
            "use_offset": True,
            "merge_pitch_and_beat": False,
            "use_program": False,
            "use_instrument": True,
            "ignored_track_names": [f"Layers{i}" for i in range(0, 8)],
            "separate_drum_pitch": True,
            "use_drum_duration": False,
            "use_durations": True,
            "durations": [
                Fraction(1, 32),
                Fraction(1, 16),
                Fraction(1, 8),
                Fraction(1, 4),
                Fraction(1, 2),
                Fraction(1, 1),
                Fraction(2, 1),
                Fraction(4, 1),
            ],
            "fold_event_attributes": False,
        }

        USE_RANDOM_SHIFT = False
        tokenizer = Tokenizer(tokenizer_config)

        model_config = {
            "hidden_size": 768,
            "n_heads": 12,
            "feed_forward_size": 4 * 768,
            "n_layers": 12,
            "tokenizer_config": tokenizer_config,
            "norm_first": False,
            "enforce_constraint_in_forward": True,
            "activation": "gelu",
            "dropout": 0.1,
            "use_mlm": False,
        }

        training_wrapper = TrainingWrapper(
            model_config=model_config,
            learning_rate=1e-4 if model_config["hidden_size"] == 512 else 1e-4,
            learning_rate_gamma=0.99,
            lr_steps_per_epoch=2836,
            # masking_scheme="ratio_superposition_mixed_h_mixed_s_w_shared_rate_&_autoregression",
            masking_scheme="simple_superposition_x**1/4",
            loss = "cross_entropy",
            use_weight_decay=True,
            warmup_steps=1000,
            collapse_inactive_events=True,
        )

    else:
        training_wrapper = TrainingWrapper.load_from_checkpoint(checkpoint)
        tokenizer_config = training_wrapper.model.tokenizer.config
        tokenizer = Tokenizer(tokenizer_config)
        tag_list = tokenizer_config["tags"]
        USE_RANDOM_SHIFT = False

    mmd_4bar_filter_fn = lambda x: f"n_bars={N_BARS}" in x

    # if a track has program 0 and is not a drum track and does not contain the word "piano" in the name, filter out the whole midi
    # we can't risk having mislabelled tracks in the dataset
    sm_filter_fn = lambda sm: not any(
        track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
        for track in sm.tracks
    )

    val_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/val_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        min_notes=4 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
        use_random_shift=USE_RANDOM_SHIFT,
        sm_filter_fn=sm_filter_fn,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Loaded {len(val_ds)} validation records")

    trn_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/trn_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        transposition_range=[-6, 6]
        if DATASET == "mmd_loops" or DATASET == "harmonic"
        else None,
        min_notes=4 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
        use_random_shift=USE_RANDOM_SHIFT,
        sm_filter_fn=sm_filter_fn,
    )
    # print len of dataset
    print(f"Loaded {len(trn_ds)} training records")

    # desert capy uses batch size 80
    trn_dl = torch.utils.data.DataLoader(
        trn_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    wandb_logger = WandbLogger(
        log_model=False,
        project="slm",
    )
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=devices,
        precision="16-mixed",
        max_epochs=150,
        log_every_n_steps=1,
        # val_check_interval=10,
        callbacks=[
            # batch size finder
            progress_bar_callback,
            # learning rate monitor
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/{name}/",
                monitor="val/loss_epoch",
                mode="min",
                save_top_k=10,
                save_last=True,
                filename="{epoch}-{step}-{val/loss_epoch:.5f}",
                every_n_epochs=1 if DATASET == "mmd_loops" else 100,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/{name}/every25",
                save_last=True,
                filename="{epoch}-{step}-{val/accuracy@1:.5f}",
                every_n_epochs=10,
                save_top_k=-1,
            ),
        ],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4,
        check_val_every_n_epoch=1 if DATASET == "mmd_loops" else 10,
    )

    if checkpoint is not None:
        trainer.fit(training_wrapper, trn_dl, val_dl, ckpt_path=checkpoint)
    else:
        trainer.fit(
            training_wrapper,
            trn_dl,
            val_dl,
        )
