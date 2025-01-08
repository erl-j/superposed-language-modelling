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
    mixed_superposition
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

        # assert that masking_scheme is compatible with use_mlm
        if self.use_mlm:
            assert self.masking_scheme == "mlm" or self.masking_scheme == "mdlm"
        else:
            assert (
                self.masking_scheme == "variable_superposition"
                or self.masking_scheme == "mml"
                or self.masking_scheme == "variable_superposition_x**1/2"
                or self.masking_scheme == "variable_superposition_x**1/4"
                or self.masking_scheme == "mixed_superposition"
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
        target_token_ids = batch["token_ids"]
        # one hot encode token_ids with model dtype
        x = torch.nn.functional.one_hot(
            token_ids, num_classes=len(self.tokenizer.vocab)
        ).to(self.get_model_dtype())
        # apply masking scheme
        if self.use_mlm:
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
            else:
                x_ta = einops.rearrange(x, "b t a v -> b (t a) v")
                if self.masking_scheme == "mml":
                    x_masked_ta = random_add_masking_mml(x_ta)
                elif self.masking_scheme == "variable_superposition":
                    x_masked_ta = random_add_masking_variable_superposition(x_ta)
                elif self.masking_scheme == "variable_superposition_x**1/2":
                    x_masked_ta = random_add_masking_variable_superposition(
                        x_ta, lambda x: x ** 0.5
                    )
                elif self.masking_scheme == "variable_superposition_x**1/4":
                    x_masked_ta = random_add_masking_variable_superposition(
                        x_ta, lambda x: x ** 0.25
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
            x_masked = x_masked * (
                self.syntax_mask[None, None, ...].to(self.get_model_dtype())
            ).to(self.get_model_device())
            # renormalize
            x_input = x_masked / x_masked.sum(dim=-1, keepdim=True)
            if self.collapse_inactive_events:
                self.model.tokenizer.collapse_undefined_attributes(x_input)
        logits = self.model(x_input)
        
        # get metrics
        metrics = self.compute_metrics(
            target_token_ids, logits, batch["n_loops_in_parent_song"]
        )

        if not self.use_mlm and not self.model.enforce_constraint_in_forward:
            # apply constraint to logits
            with torch.no_grad():
                filtered_logits = logits.clone().detach()
                zero = torch.zeros_like(x_input, dtype=x_input.dtype, device=x_input.device)
                filtered_logits[x_input.isclose(zero, rtol=1e-5)] = torch.finfo(
                    logits.dtype
                ).min
                filtered_logits_metrics = self.compute_metrics(
                    target_token_ids, filtered_logits, batch["n_loops_in_parent_song"]
                )
                # add filtered logits metrics to metrics
                for key in filtered_logits_metrics:
                    metrics[f"filtered_logits/{key}"] = filtered_logits_metrics[key]
        return metrics
    
    def step_mdlm(self, batch, batch_idx):
        token_ids = batch["token_ids"]
        target_token_ids = batch["token_ids"]
        batch_size = token_ids.shape[0]
        time = torch.rand(batch_size, 1, 1, device=self.get_model_device())
        noise = LogLinearNoise()
        total_noise, rate_noise = noise(time)
        mask_probs = torch.rand(token_ids.shape, device=self.get_model_device(), dtype=self.get_model_dtype())
        mask = mask_probs < total_noise[:, None, None]
        # where mask is True, replace token_ids with mask token
        masked_input = token_ids.clone()
        masked_input[mask] = len(self.tokenizer.vocab)
        masked_tokens_1h = torch.nn.functional.one_hot(
            token_ids, num_classes=len(self.tokenizer.vocab)+1, dtype=self.get_model_dtype(), device=self.get_model_device()
        )
        log_probs = self.model(masked_tokens_1h)
        # flatten logits and target_token_ids
        log_probs_flat = einops.rearrange(log_probs, "b e a v -> b (e a) v")
        target_token_ids_flat = einops.rearrange(target_token_ids, "b e a -> b (e a)")
        loss = rate_noise[:,None] * F.cross_entropy(log_probs_flat, target_token_ids_flat, reduction="none")
        metrics = {
            "cross_entropy": loss.mean(),
            "rate_noise": rate_noise.mean(),
            "total_noise": total_noise.mean(),
        }
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
        loss = metrics["cross_entropy"]
        # plot loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        return self.stage_step(batch, batch_idx, "trn")

    def validation_step(self, batch, batch_idx):
        self.eval()
        return self.stage_step(batch, batch_idx, "val")

    def compute_metrics(self, target_token_ids, logits, n_loops_in_parent_song):
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
                {"entropy": entropy.mean(),
                "probability": target_probs.mean(),}
            )

            # Calculate accuracies for k=[1,2,4]
            topk_values = [1, 2, 4]
            for k in topk_values:
                decoder_output_probs_sort = torch.argsort(
                    probs, dim=-1, descending=True
                )  # (batch_size, events, attributes, vocab_size)
                accuracy = (
                    (target_token_ids.unsqueeze(-1) == decoder_output_probs_sort[..., :k])
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
                metrics[f"probability/{attribute}"] = target_probs[:, :, attr_idx].mean()

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
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda
            ),
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
            bias=False
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


if __name__ == "__main__":
    SEED = 0

    torch.manual_seed(SEED)

    DATASET = "mmd_loops"

    #BATCH_SIZE = 80
    # BATCH_SIZE = 40
    BATCH_SIZE = 60

    tag_list = open(f"./data/{DATASET}/tags.txt").read().splitlines()

    N_BARS = 4 if DATASET == "harmonic" else 4

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
        "durations": [Fraction(1, 32), Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(2, 1), Fraction(4, 1)],
        "fold_event_attributes": False,
    }

    USE_RANDOM_SHIFT = False
    tokenizer = Tokenizer(tokenizer_config)

    model_config = {
        "hidden_size":768,
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
        masking_scheme="mixed_superposition",
        use_weight_decay=True,
        warmup_steps=1000,
        collapse_inactive_events=True,
    )

    # training_wrapper = TrainingWrapper.load_from_checkpoint(
    #     "./checkpoints/toasty-bush-529/last.ckpt",
    #     map_location="cpu",
    # )

    # training_wrapper.convert_mlm_to_slm()
    # training_wrapper.enforce_constraint_in_forward = False
    # training_wrapper.model.embedding_layer.weight.requires_grad = False
    # training_wrapper.model.unembedding_layer.weight.requires_grad = False

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

    # val_ds2 = MidiDataset(
    #     cache_path=f"./data/harmonic/val_midi_records_unique_pr.pt",
    #     path_filter_fn=None,
    #     genre_list=tag_list,
    #     tokenizer=tokenizer,
    #     min_notes=4 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
    #     max_notes=tokenizer_config["max_notes"],
    #     use_random_shift=USE_RANDOM_SHIFT,
    #     sm_filter_fn=sm_filter_fn,
    # )

    # val_dl2 = torch.utils.data.DataLoader(
    #     val_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    # )

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
        devices=[6,7],
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

    trainer.fit(
        training_wrapper,
        trn_dl,
        val_dl,
    )