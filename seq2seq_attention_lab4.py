"""
Seq2Seq with Bahdanau Attention — Full Implementation
======================================================
Encoder-Decoder architecture for sequence-to-sequence tasks.
Demonstrated on a toy number-word translation task:
    "1 2 3"  →  "one two three"

Architecture
------------
  Encoder   : Bidirectional GRU
  Attention : Bahdanau (additive) attention
  Decoder   : GRU + Attention context vector

Usage:
    python seq2seq_attention.py
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────
# 1.  HYPERPARAMETERS
# ─────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM   = 64
HIDDEN_DIM  = 128
DROPOUT     = 0.3
BATCH_SIZE  = 32
EPOCHS      = 80
LR          = 0.001
CLIP        = 1.0
TEACHER_FORCING_RATIO = 0.5

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"


# ─────────────────────────────────────────
# 2.  TOY DATASET  (digit → word translation)
# ─────────────────────────────────────────
NUM2WORD = {
    "0": "zero",  "1": "one",   "2": "two",   "3": "three",
    "4": "four",  "5": "five",  "6": "six",   "7": "seven",
    "8": "eight", "9": "nine",
}

def make_pair(length=4):
    digits = [str(random.randint(0, 9)) for _ in range(random.randint(1, length))]
    src    = digits
    tgt    = [NUM2WORD[d] for d in digits]
    return src, tgt

def make_dataset(n=3000):
    return [make_pair() for _ in range(n)]

raw_data  = make_dataset(3000)
split     = int(0.9 * len(raw_data))
train_raw = raw_data[:split]
valid_raw = raw_data[split:]


# ─────────────────────────────────────────
# 3.  VOCABULARY
# ─────────────────────────────────────────
class Vocab:
    def __init__(self, specials=(PAD, SOS, EOS, UNK)):
        self.tok2idx = {}
        self.idx2tok = []
        for s in specials:
            self._add(s)

    def _add(self, token):
        if token not in self.tok2idx:
            self.tok2idx[token] = len(self.idx2tok)
            self.idx2tok.append(token)

    def build(self, sentences):
        for sent in sentences:
            for tok in sent:
                self._add(tok)
        return self

    def encode(self, tokens, add_sos=False, add_eos=True):
        seq = []
        if add_sos:
            seq.append(self.tok2idx[SOS])
        seq += [self.tok2idx.get(t, self.tok2idx[UNK]) for t in tokens]
        if add_eos:
            seq.append(self.tok2idx[EOS])
        return seq

    def decode(self, indices):
        out = []
        for i in indices:
            tok = self.idx2tok[i]
            if tok in (PAD, SOS):
                continue
            if tok == EOS:
                break
            out.append(tok)
        return out

    def __len__(self):
        return len(self.idx2tok)


src_vocab = Vocab().build([s for s, _ in raw_data])
tgt_vocab = Vocab().build([t for _, t in raw_data])


# ─────────────────────────────────────────
# 4.  DATASET / DATALOADER
# ─────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids  = src_vocab.encode(src, add_sos=False, add_eos=True)
        tgt_ids  = tgt_vocab.encode(tgt, add_sos=True,  add_eos=True)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True,
                                     padding_value=src_vocab.tok2idx[PAD])
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True,
                                     padding_value=tgt_vocab.tok2idx[PAD])
    return srcs, tgts


train_loader = DataLoader(TranslationDataset(train_raw),
                          batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
valid_loader = DataLoader(TranslationDataset(valid_raw),
                          batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# ─────────────────────────────────────────
# 5.  MODEL COMPONENTS
# ─────────────────────────────────────────

# ── 5a. Encoder ──────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn     = nn.GRU(embed_dim, hidden_dim, batch_first=True,
                              bidirectional=True)
        self.fc      = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (B, T_src)
        embedded = self.dropout(self.embed(src))          # (B, T, E)
        outputs, hidden = self.rnn(embedded)
        # outputs : (B, T, 2*H)  — all timestep outputs
        # hidden  : (2, B, H)    — [fwd; bwd] last hidden states

        # Merge bidirectional final hidden → (B, H)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  # (1, B, H)
        return outputs, hidden


# ── 5b. Bahdanau Attention ───────────────
class BahdanauAttention(nn.Module):
    """
    e_t = v^T tanh(W1 * h_dec + W2 * h_enc)
    a_t = softmax(e_t)
    """
    def __init__(self, hidden_dim, enc_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(enc_dim,    hidden_dim, bias=False)
        self.v  = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden  : (B, H)
        # enc_outputs : (B, T_src, enc_dim)
        T = enc_outputs.size(1)

        dec_expanded = dec_hidden.unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        energy = torch.tanh(self.W1(dec_expanded) + self.W2(enc_outputs))  # (B, T, H)
        scores = self.v(energy).squeeze(-1)                        # (B, T)
        attn_weights = F.softmax(scores, dim=1)                    # (B, T)

        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_dim)
        return context, attn_weights


# ── 5c. Decoder ──────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_dim, dropout):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, enc_dim)
        self.rnn       = nn.GRU(embed_dim + enc_dim, hidden_dim, batch_first=True)
        self.fc_out    = nn.Linear(hidden_dim + enc_dim + embed_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward_step(self, token, hidden, enc_outputs):
        # token   : (B,)
        # hidden  : (1, B, H)
        embedded = self.dropout(self.embed(token.unsqueeze(1)))   # (B, 1, E)
        context, attn = self.attention(hidden.squeeze(0), enc_outputs)  # (B, enc_dim)
        context_exp   = context.unsqueeze(1)                       # (B, 1, enc_dim)
        rnn_input     = torch.cat([embedded, context_exp], dim=2)  # (B, 1, E+enc_dim)
        output, hidden = self.rnn(rnn_input, hidden)               # (B,1,H), (1,B,H)

        pred_input = torch.cat([
            output.squeeze(1),           # (B, H)
            context,                     # (B, enc_dim)
            embedded.squeeze(1),         # (B, E)
        ], dim=1)
        logits = self.fc_out(pred_input)  # (B, vocab_size)
        return logits, hidden, attn

    def forward(self, tgt, hidden, enc_outputs, teacher_forcing_ratio=0.5):
        # tgt: (B, T_tgt)
        B, T = tgt.size()
        vocab_size = self.fc_out.out_features
        outputs    = torch.zeros(B, T, vocab_size).to(tgt.device)
        attentions = []

        token = tgt[:, 0]  # <sos>
        for t in range(1, T):
            logits, hidden, attn = self.forward_step(token, hidden, enc_outputs)
            outputs[:, t, :] = logits
            attentions.append(attn)
            use_teacher = random.random() < teacher_forcing_ratio
            token = tgt[:, t] if use_teacher else logits.argmax(1)

        return outputs, torch.stack(attentions, dim=1)


# ── 5d. Seq2Seq wrapper ──────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        enc_outputs, hidden = self.encoder(src)
        outputs, attentions = self.decoder(tgt, hidden, enc_outputs,
                                           teacher_forcing_ratio)
        return outputs, attentions


# ─────────────────────────────────────────
# 6.  INSTANTIATE
# ─────────────────────────────────────────
ENC_OUT_DIM = HIDDEN_DIM * 2   # bidirectional

encoder = Encoder(len(src_vocab), EMBED_DIM, HIDDEN_DIM, DROPOUT)
decoder = Decoder(len(tgt_vocab), EMBED_DIM, HIDDEN_DIM, ENC_OUT_DIM, DROPOUT)
model   = Seq2Seq(encoder, decoder).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.tok2idx[PAD])

print(f"Device      : {DEVICE}")
print(f"Src vocab   : {len(src_vocab)}")
print(f"Tgt vocab   : {len(tgt_vocab)}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters  : {total_params:,}\n")


# ─────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────
def run_epoch(loader, train=True, tf_ratio=0.5):
    model.train() if train else model.eval()
    total_loss = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs, _ = model(src, tgt, tf_ratio if train else 0.0)
            # outputs: (B, T, V), skip index 0 (SOS position)
            loss = criterion(
                outputs[:, 1:, :].contiguous().view(-1, len(tgt_vocab)),
                tgt[:, 1:].contiguous().view(-1)
            )
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True,  tf_ratio=TEACHER_FORCING_RATIO)
    valid_loss = run_epoch(valid_loader, train=False)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train: {train_loss:.4f}  Valid: {valid_loss:.4f}")


# ─────────────────────────────────────────
# 8.  INFERENCE
# ─────────────────────────────────────────
def translate(src_tokens, max_len=20):
    """Greedy decoding with attention."""
    model.eval()
    src_ids = torch.tensor(
        [src_vocab.encode(src_tokens, add_sos=False, add_eos=True)]
    ).to(DEVICE)

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(src_ids)

        token = torch.tensor([tgt_vocab.tok2idx[SOS]]).to(DEVICE)
        decoded, attn_weights = [], []

        for _ in range(max_len):
            logits, hidden, attn = model.decoder.forward_step(token, hidden, enc_outputs)
            token = logits.argmax(1)
            attn_weights.append(attn.squeeze(0).cpu().tolist())
            decoded.append(token.item())
            if token.item() == tgt_vocab.tok2idx[EOS]:
                break

    return tgt_vocab.decode(decoded), attn_weights


# ─────────────────────────────────────────
# 9.  EVALUATION
# ─────────────────────────────────────────
print("\n" + "="*55)
print("SAMPLE TRANSLATIONS")
print("="*55)
test_cases = [
    ["3", "1", "4"],
    ["0", "9"],
    ["7", "7", "7"],
    ["1", "2", "3", "4"],
    ["5", "0", "5"],
]
for src in test_cases:
    pred, _ = translate(src)
    expected = [NUM2WORD[d] for d in src]
    status   = "✓" if pred == expected else "✗"
    print(f"  {status}  {' '.join(src):12s}  →  {' '.join(pred)}")
    if pred != expected:
        print(f"       expected: {' '.join(expected)}")


# ─────────────────────────────────────────
# 10. ATTENTION VISUALIZATION (text)
# ─────────────────────────────────────────
def print_attention(src, pred, attn_weights):
    print("\nAttention matrix (rows=tgt, cols=src):")
    src_labels = src + [EOS]
    print("         " + "  ".join(f"{s:>6}" for s in src_labels))
    for i, (tgt_tok, row) in enumerate(zip(pred, attn_weights)):
        bar = "  ".join(f"{v:6.3f}" for v in row[:len(src_labels)])
        print(f"  {tgt_tok:>6}  {bar}")


src_example = ["2", "0", "2", "6"]
pred, attn  = translate(src_example)
print_attention(src_example, pred, attn)
