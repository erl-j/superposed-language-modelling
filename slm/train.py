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
from tokenizer import Tokenizer
from torch import nn
from augmentation import transpose_sm


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
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()
        vocab_size = len(vocab)
        self.tokenizer = Tokenizer(tokenizer_config)
        self.format_mask = self.tokenizer.get_format_mask()
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
        
        self.mask = get_mask(max_seq_len)
       
        # save fig of mask
        # plt.imshow(get_mask(4))
        # plt.savefig("artefacts/mask.png")
        self.seq_len = max_seq_len

    
    def forward(self, cnst, seq):
        cnst = cnst*self.format_mask[None,...].to(self.device)
        decoder_len = seq.shape[1]
        # pad decoder tokens with zeros
        seq = F.pad(seq, (0, 0, 0, self.seq_len-seq.shape[1]))
        # concatenate cnst_and_seq and seq
        cnst_and_seq = torch.cat([cnst, seq], dim=1)
        # assert cnst_and_seq.shape[1] == self.seq_len*2
        ze = self.embedding_layer(cnst_and_seq)
        pos = self.positional_encoding[:, :2*self.seq_len, :].to(self.device)
        # repeat
        # add positional encoding
        ze = ze + pos
        # pass through transformer
        zl = self.transformer(ze, mask=self.mask.to(self.device))
        zd = zl[:, self.seq_len:, :]
        # assert zd.shape[1] == self.seq_len
        decoder_logits = self.decoder_output_layer(zd)
        # assert decoder_logits.shape[1] == self.seq_len
        # # to ensure the encoder constraint is respected
        # # multiply decoder_logits by right shifted encoder fts
        # # overlap between encoder and decoder
        # remove -inf from decoder logits
        decoder_logits[:,:-1]= decoder_logits[:,:-1] + (1-cnst[:,1:,:])*1e9
        # crop to decoder length
        decoder_logits = decoder_logits[:, :decoder_len, :]
        return decoder_logits
    

    def generate(
        self, encoder_ft, temperature=1.0, max_len=1000, top_p=0.9, top_k=0
    ):
        """
        Does not use KV caching so it's slow
        """
        # make copies
        encoder_ft = encoder_ft.clone()
        decoder_ft = encoder_ft.clone()[:, :1, :]
        
        while decoder_ft.shape[1] < max_len:
            decoder_logits = self.forward(encoder_ft, decoder_ft)[:, -1, :]
            decoder_probs = F.softmax(decoder_logits / temperature, dim=-1)
            # top k
            if top_k > 0:
                # mask
                decoder_probs[decoder_probs < torch.topk(decoder_probs, top_k)[0][..., -1:]] = 0
                # renormalize
                decoder_probs = decoder_probs / decoder_probs.sum(dim=-1, keepdim=True)
          
            sampled_token = torch.multinomial(decoder_probs, num_samples=1)
            # one hot
            sampled_token = torch.zeros_like(decoder_probs).scatter_(1, sampled_token, 1)
            decoder_ft = torch.cat(
                [decoder_ft, sampled_token[:,None,:]], dim=1
            )
        return decoder_ft

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

        decoder_output_tokens = decoder_input[:, 1:]
        decoder_output_logits = decoder_output_logits[:, :-1]

        decoder_output_tokens_index = torch.argmax(decoder_output_tokens, dim=-1)

        ce = F.cross_entropy(
            decoder_output_logits.reshape(-1, decoder_output_logits.shape[-1]),
            decoder_output_tokens_index.reshape(-1),
        )
        metrics = {}
        metrics["cross_entropy"] = ce
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
        loss = metrics["cross_entropy"]
        self.log("trn/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
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
        loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
        "max_notes":100 * N_BARS,
        "min_tempo":50,
        "max_tempo":200,
        "n_tempo_bins": 16,
        "time_signatures": None,
        "tags": genre_list,
        "shuffle_notes": True,
        "use_offset": True,
        "merge_pitch_and_beat":True,
        "use_program": True,
        "ignored_track_names":[f"Layers{i}" for i in range(0, 8)],
    }

    tokenizer = Tokenizer(
        tokenizer_config
    )

    trn_ds = MidiDataset(
        cache_path="./artefacts/trn_midi_records.pt",
        path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4],
    )

    val_ds = MidiDataset(
        cache_path="./artefacts/val_midi_records.pt",
        path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
    )
  
    BATCH_SIZE = 2

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
        n_heads=8,
        feed_forward_size=4*768,
        n_layers=10,
        vocab = tokenizer.vocab,
        max_seq_len=tokenizer.total_len,
        learning_rate=1e-4,
        tokenizer_config=tokenizer_config,
        sliding_mask=True,
    )

    wandb_logger = WandbLogger(log_model="all", project="slm")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(accelerator="gpu",
    devices=[3],
    precision=32,
    max_epochs=None,
    log_every_n_steps=1,
    # val_check_interval=10,
    callbacks=[progress_bar_callback,
            pl.callbacks.ModelCheckpoint(
            dirpath=f"./checkpoints/{name}/",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{step}-{val/loss:.2f}-{trn/loss:.2f}",
            train_time_interval = datetime.timedelta(minutes=30),)],
    logger=wandb_logger,
    accumulate_grad_batches=4
    )

    trainer.fit(model,
     trn_dl,
     val_dl,
    )