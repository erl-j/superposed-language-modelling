import datetime
import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from data import MidiDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from merged_tokenizer_2 import MergedTokenizer2 as MergedTokenizer
from torch import nn
from augmentation import transpose_sm
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering

class DecoderOnlyModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_layers,
        vocab,
        max_seq_len,
        learning_rate,
        tokenizer_config,
        sliding_mask=False,
        one_hot_input=False,
        normalize_by_masking_ratio=False,
        learning_rate_gamma=0.9,
        note_decoder_layers=2,
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()
        vocab_size = len(vocab)
        self.tokenizer = MergedTokenizer(tokenizer_config)
        self.format_mask = torch.Tensor(self.tokenizer.get_format_mask())
        self.vocab = vocab
        # intialize positional encoding. one per step in sequence
        self.positional_encoding  = nn.Parameter(torch.zeros(1, max_seq_len*2, hidden_size), requires_grad=True)
        self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=False)
        self.one_hot_input = one_hot_input
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=feed_forward_size,
            dropout=0.1,
            batch_first=True,
            ),
            num_layers=n_layers,
        )

        self.note_decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=feed_forward_size,
            dropout=0.1,
            ),
            num_layers=note_decoder_layers,
        )

        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size)

        def get_mask(seq_len):
            m = torch.ones(seq_len*2, seq_len*2, dtype=torch.bool) * float("-inf")
            # give upper left triangle
            m[:seq_len, :seq_len] = 0
            for i in range(0, seq_len):
                if sliding_mask:
                    m[i+seq_len, i+1:i+seq_len+1] = 0
                else:
                    m[i+seq_len, 0:i+seq_len+1] = 0
            return m
        
        self.mask = get_mask(self.tokenizer.config["max_notes"])
       
        # save fig of mask
        # plt.imshow(get_mask(10))
        # plt.savefig("artefacts/mask.png")
        self.seq_len = max_seq_len

        self.n_attributes = len(self.tokenizer.note_attribute_order)

        self.attribute_embedding = nn.Embedding(self.n_attributes, hidden_size)

        self.attribute_mask = torch.nn.Transformer.generate_square_subsequent_mask(len(self.tokenizer.note_attribute_order))

        self.normalize_by_masking_ratio = normalize_by_masking_ratio

        self.learning_rate_gamma = learning_rate_gamma

    def note_forward(self, con, tgt):

        batch, time_attr, ft = con.shape

        con = con * self.format_mask[None, ...].to(self.device)
        tgt = tgt * self.format_mask[None, ...].to(self.device)

        con = einops.rearrange(con, "b (t a) ft -> b t a ft", a=self.n_attributes)
        tgt = einops.rearrange(tgt, "b (t a) ft -> b t a ft", a=self.n_attributes)

        n_notes = con.shape[1]


        con_and_tgt = torch.cat([con, tgt], dim=1)
        # embed
        ze = self.embedding_layer(con_and_tgt)
        ze = ze.sum(dim=2)
        pos = self.positional_encoding[:, :ze.shape[1], :].to(self.device)
        ze = ze + pos
        # pass through transformer
        zl = self.transformer(ze, mask=self.mask.to(self.device))
        # get output part
        note_z = zl[:, n_notes-1:-1, :]

        return note_z
    
    def attribute_forward(self, con, tgt, note_z):

        con = con * self.format_mask[None, ...].to(self.device)
        tgt = tgt * self.format_mask[None, ...].to(self.device)

        con_flat = con

        # reshape
        con = einops.rearrange(con, "b (t a) ft -> b t a ft", a=self.n_attributes)
        tgt = einops.rearrange(tgt, "b (t a) ft -> b t a ft", a=self.n_attributes)

        n_notes = con.shape[1]

        # note embeddings
        note_z = einops.rearrange(note_z, "b t ft -> (b t) 1 ft")

        attr_tgt = einops.rearrange(tgt, "b t a ft -> (b t) a ft", a=self.n_attributes)

        attr_z = self.embedding_layer(attr_tgt)

        # pad in_tgt with zero at start of attribute axis
        attr_z = F.pad(attr_z, (0, 0, 1, 0))
        # remove last frame
        attr_z = attr_z[:, :-1, :]

        # add attribute embedding
        attr_z = attr_z + self.attribute_embedding.weight[None, :, :]

        # run through decoder
        zd = self.note_decoder(
            attr_z,
            note_z,
            tgt_is_causal=True,
            tgt_mask=self.attribute_mask.to(self.device),
        )

        zd = einops.rearrange(
            zd, "(b t) a ft -> b (t a) ft", t=n_notes, ft=zd.shape[-1]
        )

        decoder_logits = self.decoder_output_layer(zd)

        decoder_logits = decoder_logits - (1 - con_flat) * 1e5

        return decoder_logits

    def forward(self, con, tgt):

        # TODO: No padding currently, make sure it's not needed
        # get shape
        batch, time_attr, ft = con.shape

        con = con * self.format_mask[None, ...].to(self.device)
        tgt = tgt * self.format_mask[None, ...].to(self.device)

        con_flat = con

        # reshape
        con = einops.rearrange(con, "b (t a) ft -> b t a ft", a=self.n_attributes)
        tgt = einops.rearrange(tgt, "b (t a) ft -> b t a ft", a=self.n_attributes)

        n_notes = con.shape[1]


        con_and_tgt = torch.cat([con, tgt], dim=1)
        # embed
        ze = self.embedding_layer(con_and_tgt)
        ze = ze.sum(dim=2)
        pos = self.positional_encoding[:, :ze.shape[1], :].to(self.device)
        ze = ze + pos
        # pass through transformer
        zl = self.transformer(ze, mask=self.mask.to(self.device))
        # get output part
        note_z = zl[:, n_notes-1:-1, :]

        # note embeddings
        note_z = einops.rearrange(note_z, "b t ft -> (b t) 1 ft")

        attr_tgt= einops.rearrange(tgt, "b t a ft -> (b t) a ft", a=self.n_attributes)

        attr_z = self.embedding_layer(attr_tgt)

        # pad in_tgt with zero at start of attribute axis
        attr_z = F.pad(attr_z, (0, 0, 1, 0))
        # remove last frame
        attr_z = attr_z[:, :-1, :]

        # add attribute embedding
        attr_z = attr_z + self.attribute_embedding.weight[None, :, :]


        # run through decoder
        zd = self.note_decoder(attr_z, note_z, tgt_is_causal=True, tgt_mask=self.attribute_mask.to(self.device))

        zd = einops.rearrange(zd, "(b t) a ft -> b (t a) ft", t=n_notes, ft=zd.shape[-1])

        decoder_logits = self.decoder_output_layer(zd)
      
        decoder_logits = decoder_logits - (1-con_flat)*1e5
        # crop to decoder length
        return decoder_logits
    

    def generate(
        self, con, temperature=1.0, max_len=1000, top_p=1, top_k=0
    ):
        """
        Does not use KV caching so it's slow
        """
        # make copies
        seq = con.clone()

        batch, time, ft = seq.shape
        
        self.eval()
        with torch.no_grad():
            for note_idx in tqdm(range(self.tokenizer.config["max_notes"])):
                    note_z = self.note_forward(con, seq)
                    for attribute_idx in range(self.n_attributes):
                        decoder_logits = self.attribute_forward(con, seq, note_z)

                        decoder_logits = einops.rearrange(decoder_logits, "b s v -> (b s) v", b=batch)

                        decoder_logits = top_k_top_p_filtering(decoder_logits, top_k=top_k, top_p=top_p)

                        decoder_logits = einops.rearrange(decoder_logits, "(b s) v -> b s v", b=batch)

                        decoder_probs = F.softmax(decoder_logits / temperature, dim=-1)
                        # top k

                        
                        sampled_token = torch.multinomial(decoder_probs[0], num_samples=1)[None,...]
                        # one hot
                        sampled_token = torch.nn.functional.one_hot(sampled_token, num_classes=len(self.vocab))
                        # add sampled token to sequence
                        seq[:, note_idx*self.n_attributes + attribute_idx, :] = sampled_token[:, note_idx*self.n_attributes + attribute_idx, :]
        return seq

    def step(self, batch, batch_idx):
        if self.one_hot_input:
            x = batch
        else:
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()
        
        batch_size = x.shape[0]

        # create masking ratios
        masking_ratios = torch.rand(batch_size, device=self.device)
        mask = torch.rand_like(x, device=self.device)<masking_ratios[:,None,None]
        # mask encoder input
        # encoder input is mask or token tensor with undefined
        encoder_input = torch.clamp(x + mask, 0, 1)

        decoder_input = x
        decoder_output_logits = self(encoder_input, decoder_input)

        decoder_output_tokens = decoder_input
        decoder_output_logits = decoder_output_logits

        decoder_output_tokens_index = torch.argmax(decoder_output_tokens, dim=-1)

        ce = F.cross_entropy(
            decoder_output_logits.reshape(-1, decoder_output_logits.shape[-1]),
            decoder_output_tokens_index.reshape(-1),
            reduction = "none",
        )
        # reshape to batch, loss
        ce = ce.reshape(batch_size, -1)

        norm_ce = (ce.mean(dim=-1) / masking_ratios).mean()

        metrics = {}
        metrics["cross_entropy"] = ce.mean()
        metrics["cross_entropy_normalized"] = norm_ce
        # TODO: check that this code is correct
        with torch.no_grad():
            # get probability of the correct token
            decoder_output_probs = F.softmax(decoder_output_logits, dim=-1)
            probability = torch.gather(
                decoder_output_probs, dim=-1, index=decoder_output_tokens_index.unsqueeze(-1)
            ).squeeze(-1)
            metrics["probability"] = probability.mean()
            # sort yhat by probability
            decoder_output_probs_sort = torch.argsort(
                decoder_output_probs, dim=-1, descending=True
            )
            for k in [1, 2, 4]:
                metrics[f"accuracy@{k}"] = (
                    (
                        decoder_output_tokens_index.unsqueeze(-1)
                        == decoder_output_probs_sort[:, :, :k]
                    )
                    .any(dim=-1)
                    .float()
                    .mean()
                )
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"trn/{metric}", metrics[metric], prog_bar=True)
        if self.normalize_by_masking_ratio:
            loss = metrics["cross_entropy_normalized"]
        else:
            loss = metrics["cross_entropy"]
        self.log("trn/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # log wandb name
        self.log("gpu", loss.device.index)
        return loss

    def validation_step(self, batch, batch_idx):
        if False and batch_idx == 0:
            if self.one_hot_input:
                x = batch
            else:
                x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()
            encoder_input = x
            decoder_input = x
            generated = self.generate(
                encoder_input*0+1, decoder_input[:,:4], temperature=1.0, max_len=50
            )
            generated = torch.argmax(generated, dim=-1)
            generated = generated.cpu().numpy()
            generated = generated[0]
            # decode with vocab
            generated = [self.vocab[i] for i in generated]
             # write to file
            print(generated)
            with open("artefacts/generated.txt", "w") as f:
                f.write("\n".join(generated))

        with torch.no_grad():
            metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"val/{metric}", metrics[metric], prog_bar=True, on_step=True, on_epoch=True)
        if self.normalize_by_masking_ratio:
            loss = metrics["cross_entropy_normalized"]
        else:
            loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.learning_rate_gamma
        )
        return [optimizer], [scheduler]

if __name__ == "__main__":

    genre_list = [
    "other",
    "pop",
    "rock",
    "italian%2cfrench%2cspanish",
    "classical",
    "romantic",
    "renaissance",
    "alternative-indie",
    "metal",
    "traditional",
    "country",
    "baroque",
    "punk",
    "modern",
    "jazz",
    "dance-eletric",
    "rnb-soul",
    "medley",
    "blues",
    "hip-hop-rap",
    "hits of the 2000s",
    "instrumental",
    "midi karaoke",
    "folk",
    "newage",
    "latino",
    "hits of the 1980s",
    "hits of 2011 2020",
    "musical%2cfilm%2ctv",
    "reggae-ska",
    "hits of the 1970s",
    "christian-gospel",
    "world",
    "early_20th_century",
    "hits of the 1990s",
    "grunge",
    "australian artists",
    "funk",
    "best of british"
    ]

    N_BARS = 4

    tokenizer_config = {
        "ticks_per_beat":24,
        "pitch_range":[0, 128],
        "max_beats":4*N_BARS,
        "max_notes":75 * N_BARS,
        "min_tempo":50,
        "max_tempo":200,
        "n_tempo_bins": 16,
        "n_velocity_bins": 32,
        "time_signatures": None,
        "tags": genre_list,
        "shuffle_notes": True,
        "use_offset": True,
        "merge_pitch_and_beat":False,
        "use_program": False,
        "use_instrument": True,
        "ignored_track_names":[f"Layers{i}" for i in range(0, 8)],
        "separate_drum_pitch": True,
        "use_drum_duration": False,
    }

    tokenizer = MergedTokenizer(
        tokenizer_config
    )

    trn_ds = MidiDataset(
        cache_path="./artefacts/trn_midi_records.pt",
        path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4],
        min_notes = 8*N_BARS,
        max_notes = tokenizer_config["max_notes"],
    )

    val_ds = MidiDataset(
        cache_path="./artefacts/val_midi_records.pt",
        path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
        min_notes=8 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )
  
    BATCH_SIZE = 32

    trn_dl = torch.utils.data.DataLoader(
        trn_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    model = DecoderOnlyModel(
        hidden_size=768,
        n_heads=16,
        feed_forward_size=2*768,
        n_layers=7,
        vocab = tokenizer.vocab,
        max_seq_len=tokenizer.total_len,
        learning_rate=1e-4,
        tokenizer_config=tokenizer_config,
        sliding_mask=True,
        normalize_by_masking_ratio=False,
        learning_rate_gamma=0.99,
        note_decoder_layers=1,
    )

    wandb_logger = WandbLogger(log_model="all", project="slm")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
    accelerator="gpu",
    devices=[3],
    precision=32,
    max_epochs=None,
    log_every_n_steps=1,
    # val_check_interval=10,
    callbacks=[progress_bar_callback,
            # learning rate monitor
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
            dirpath=f"./checkpoints/{name}/",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{step}-{val/loss:.2f}-{trn/loss:.2f}",
            train_time_interval = datetime.timedelta(minutes=30),)],
    logger=wandb_logger,
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,
    )

    trainer.fit(
        model,
        trn_dl,
        val_dl,
    )