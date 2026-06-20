import time
import json
import torch
from tinygpt_model import TinyGPT
from tokenizers_lib import WordTokenizer, CharTokenizer, SubwordTokenizer

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

BLOCK_SIZE = 48
BATCH_SIZE = 16
N_EMBD = 48
N_HEAD = 4
N_LAYER = 2
MAX_ITERS = 200
EVAL_INTERVAL = 50
LR = 3e-3


def get_batches(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def run_one(tokenizer, label):
    print(f"\n=== Eksperimen: {label} tokenization ===")
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    vocab_size = tokenizer.vocab_size
    print(f"Jumlah token total: {len(ids)} | Ukuran vocab: {vocab_size}")

    block_size = min(BLOCK_SIZE, len(val_data) - 1, len(train_data) - 1)
    model = TinyGPT(vocab_size, block_size, N_EMBD, N_HEAD, N_LAYER).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    start = time.time()
    for it in range(MAX_ITERS + 1):
        xb, yb = get_batches(train_data, block_size, BATCH_SIZE)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batches(val_data, block_size, BATCH_SIZE)
                _, val_loss = model(xv, yv)
            model.train()
            history.append({"iter": it, "train_loss": loss.item(), "val_loss": val_loss.item()})
            print(f"iter {it:4d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")

    elapsed = time.time() - start

    # generate sample
    model.eval()
    context_text = "Kecerdasan buatan"
    try:
        context_ids = tokenizer.encode(context_text)
        idx = torch.tensor([context_ids], dtype=torch.long).to(device)
        out_ids = model.generate(idx, max_new_tokens=60, temperature=0.8)[0].tolist()
        generated = tokenizer.decode(out_ids)
    except Exception as e:
        generated = f"(gagal generate: {e})"

    final_val_loss = history[-1]["val_loss"]
    perplexity = float(torch.exp(torch.tensor(final_val_loss)))

    result = {
        "label": label,
        "vocab_size": vocab_size,
        "num_tokens": len(ids),
        "num_params": n_params,
        "train_time_sec": round(elapsed, 2),
        "final_train_loss": round(history[-1]["train_loss"], 4),
        "final_val_loss": round(final_val_loss, 4),
        "perplexity": round(perplexity, 2),
        "generated_sample": generated,
        "history": history,
    }
    return result


if __name__ == "__main__":
    results = []

    word_tok = WordTokenizer(text)
    results.append(run_one(word_tok, "word"))

    char_tok = CharTokenizer(text)
    results.append(run_one(char_tok, "char"))

    subword_tok = SubwordTokenizer(text, vocab_size=400)
    results.append(run_one(subword_tok, "subword"))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n\n=== RINGKASAN PERBANDINGAN ===")
    print(f"{'Metode':<10}{'Vocab':<10}{'#Token':<10}{'Params':<12}{'Waktu(s)':<10}{'TrainLoss':<12}{'ValLoss':<10}{'PPL':<10}")
    for r in results:
        print(f"{r['label']:<10}{r['vocab_size']:<10}{r['num_tokens']:<10}{r['num_params']:<12}"
              f"{r['train_time_sec']:<10}{r['final_train_loss']:<12}{r['final_val_loss']:<10}{r['perplexity']:<10}")
