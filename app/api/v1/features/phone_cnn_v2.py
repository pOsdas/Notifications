import argparse
import json
import os
import random
from pathlib import Path
import string
from typing import List, Tuple, Dict, Any
from django.conf import settings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, "artifacts")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MAX_LEN = 48  # немного больше, чтобы покрыть расширения и префиксы
CHARS = list("0123456789+-() .:extEXTtelТел,+/")
PAD = "<PAD>"
UNK = "<UNK>"
VOCAB = [PAD, UNK] + CHARS
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

EMBED_DIM = 48
NUM_FILTERS = 128
KERNEL_SIZES = [2, 3, 4, 5]
FC_HIDDEN = 128
DEFAULT_BATCH = 512
DEFAULT_EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHAR_CNN_MODEL_PREFIX = settings.CHAR_CNN_MODEL_PREFIX
CHAR_CNN_MODEL_PREFIX = "phone_cnn_v2"


def encode_text(text: str, max_len: int = MAX_LEN) -> List[int]:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    seq = []
    for ch in text[:max_len]:
        seq.append(CHAR2IDX.get(ch, CHAR2IDX[UNK]))
    if len(seq) < max_len:
        seq += [CHAR2IDX[PAD]] * (max_len - len(seq))
    return seq


def decode_indices(idxs: List[int]) -> str:
    return "".join(IDX2CHAR.get(i, "?") for i in idxs)


def gen_phone_templates() -> List[str]:
    """Список шаблонов с X как заменителем для цифр и CC для кода страны."""
    templates = [
        "+CC XXXXXXXXXX",
        "+CC-XXX-XXX-XXXX",
        "(XXX) XXX-XXXX",
        "XXX-XXX-XXXX",
        "XXXXXXXXXX",
        "8 (XXX) XXX-XX-XX",
        "+CC (XX) XXX-XX-XX",
        "+CCXXXXXXXXXXX",
        "+CC.XXX.XXX.XXXX",
        "+CC/XX/XXXX/XXXX",
        "0XX XXX XX XX",
        "+CC XX XX XX XX",
        "00CC XXXXXXXXXX",
        "tel:+CCXXXXXXXXXX",
        "tel:XXX-XXX-XXXX",
        "+CC (0) XX XXX XXXX",
        "CC XXXXXXXXXX",
        "+CC X XX XXX XXXX",
    ]
    countries = [
        "1", "7", "44", "380", "49", "33", "39", "34", "91", "86", "61", "81",
        "55", "351", "420", "47", "46", "32", "36", "90", "52", "27", "212"
    ]
    results = []
    for c in countries:
        for t in templates:
            s = t.replace("CC", c)
            out = ""
            for ch in s:
                if ch == "X":
                    out += str(random.randint(0, 9))
                else:
                    out += ch
            results.append(out.strip())
    # ручные примеры по желанию
    # results += [
    #     "+1 800 555 0123",
    #     "+7 (912) 345-67-89",
    #     "8 912 345 67 89",
    #     "+380671234567",
    #     "+44 20 7946 0958",
    #     "+49-151-12345678",
    #     "+91-9876543210",
    #     "+86 137 1234 5678",
    # ]
    return list(set(results))


def mutate_phone(s: str) -> str:
    """Добавляем реалистичные мутации - удаление/добавление разделителей, префиксы, расширения, опечатки"""
    out = s
    # удалить случайные дефисы/пробелы
    if random.random() < 0.25:
        out = out.replace("-", " ")
    if random.random() < 0.15:
        out = out.replace(" ", "")
    # добавить 'tel:' или 'Тел:'
    if random.random() < 0.08:
        out = "tel: " + out
    if random.random() < 0.05:
        out = "Тел: " + out
    # добавить расширение
    if random.random() < 0.12:
        ext_len = random.randint(1, 4)
        out = out + " ext " + "".join(str(random.randint(0,9)) for _ in range(ext_len))
    # добавить случайную опечатку в одном месте (редко)
    if random.random() < 0.08:
        pos = random.randint(0, len(out)-1)
        if out[pos].isdigit():
            if random.random() < 0.5:
                out = out[:pos] + random.choice(string.ascii_letters) + out[pos+1:]
            else:
                out = out[:pos] + out[pos] + out[pos:]
        else:
            out = out[:pos] + random.choice("- ") + out[pos+1:]
    # вложенные скобки/разные тире иногда
    if random.random() < 0.05:
        out = out.replace("(", "[") if "(" in out else out
    return out


def random_noise_string(max_len=30) -> str:
    pool = string.ascii_letters + string.digits + "!?@#$%&*[]{}\\/|_=:;<>"
    L = random.randint(3, max_len)
    return "".join(random.choice(pool) for _ in range(L))


def generate_hard_negatives() -> List[str]:
    """Генерация ложно положительных строк, похожих на телефоны, но не являющихся ими"""
    res = []
    for _ in range(1500):
        res.append("Заказ#" + "".join(random.choices(string.digits, k=random.choice([8, 9, 10, 12]))))
        res.append("Order#" + "".join(random.choices(string.digits, k=random.choice([8, 9, 10, 12]))))

    for _ in range(3000):
        res.append("SKU-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8)))

    for _ in range(2000):
        res.append("".join(random.choices(string.digits, k=16)))

    words = ["FLOWERS", "SERVICE", "SUPPORT", "HOTLINE", "CALLME", "СЕРВИС", "ПОДДЕРЖКА", "ГОРЯЧАЯ ЛИНИЯ"]
    for w in words:
        res.append("+1-800-" + w)

    for _ in range(3000):
        s = random.choice(["tel:", "phone:", "contact:", "тел:", "контакт:"])
        s += random_noise_string(max_len=20)
        res.append(s)

    for _ in range(4000):
        chunk = " ".join("".join(
            random.choices(string.digits, k=random.choice([2, 3, 4]))
        ) for _ in range(random.randint(2, 8)))
        res.append(chunk)

    for _ in range(3000):
        res.append("user-" + "".join(random.choices(string.ascii_lowercase, k=6)) + "-" + "".join(
            random.choices(string.digits, k=6)))

    for _ in range(4000):
        user = "".join(random.choices(string.ascii_lowercase, k=random.randint(3,10)))
        dom = random.choice(["gmail.com", "example.com", "service.net", "mail.ru"])
        res.append(f"{user}@{dom}")

    return res


def generate_dataset(total_samples: int = 200000) -> Tuple[List[str], List[int]]:
    """
    Возвращает X, y списки.
    total_samples - общее число примеров (плюс/минус, сбалансировано)
    """
    assert total_samples >= 1000, "Увеличьте total_samples"
    half = total_samples // 2
    pos_templates = gen_phone_templates()
    X = []
    y = []
    # positive
    for _ in range(half):
        base = random.choice(pos_templates)
        s = mutate_phone(base)
        # иногда вставляем non-breakable space or weird space
        if random.random() < 0.02:
            s = s.replace(" ", "\u00A0")
        X.append(s)
        y.append(1)
    # negative: смесь случайного шума и "трудных"
    hard = generate_hard_negatives()
    neg_count_hard = min(len(hard), half // 2)
    # hard negatives first
    for i in range(neg_count_hard):
        X.append(hard[i])
        y.append(0)
    # rest negative: random noises, numeric garbage, uuid-like
    for _ in range(half - neg_count_hard):
        r = random.random()
        if r < 0.35:
            X.append(random_noise_string(max_len=30))
        elif r < 0.65:
            X.append("".join(random.choices(string.digits, k=random.choice([1, 2, 3, 4, 20, 24, 30]))))
        elif r < 0.85:
            X.append("ID-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10)))
        else:
            parts = []
            for _ in range(random.randint(2,6)):
                parts.append("".join(random.choices(
                    string.ascii_lowercase + string.digits, k=random.randint(1, 4)
                )))
            X.append(" ".join(parts))
        y.append(0)

    # перемешиваем dataset
    combined = list(zip(X, y))
    random.shuffle(combined)
    Xs, ys = zip(*combined)
    return list(Xs), list(ys)


class PhoneDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = torch.tensor(encode_text(self.texts[idx]), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 num_filters: int = NUM_FILTERS, kernel_sizes: List[int] = KERNEL_SIZES,
                 fc_hidden: int = FC_HIDDEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=CHAR2IDX[PAD])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        conv_output = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(conv_output, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            c = conv(emb)
            c = torch.relu(c)
            c = torch.max(c, dim=2)[0]
            conv_outs.append(c)
        cat = torch.cat(conv_outs, dim=1)
        logits = self.fc(cat).squeeze(1)
        return logits  # сырые logits


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader = None,
          epochs: int = DEFAULT_EPOCHS, lr: float = 1e-3, print_every: int = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(DEVICE)
    history = {"train_loss": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg)
        if epoch % print_every == 0:
            print(f"[Epoch {epoch}] train loss: {avg:.6f}")
        if val_loader is not None:
            metrics = evaluate(model, val_loader)
            history["val"].append(metrics)
    return model, history


def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().astype(int).tolist())
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    print(f"Eval: precision={p:.4f} recall={r:.4f} f1={f1:.4f} auc={auc:.4f}")
    return {"precision": p, "recall": r, "f1": f1, "auc": auc}


def save_artifacts(model: nn.Module, prefix: str = CHAR_CNN_MODEL_PREFIX):
    os.makedirs("artifacts", exist_ok=True)
    state_path = os.path.join("artifacts", f"{prefix}.pt")
    vocab_path = os.path.join("artifacts", f"{prefix}_vocab.json")
    torch.save(model.state_dict(), state_path)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(VOCAB, f, ensure_ascii=False)
    print("Saved state_dict to", state_path)
    print("Saved vocab to", vocab_path)
    # TorchScript export
    try:
        scripted_path = os.path.join("artifacts", f"{prefix}_scripted.pt")
        model_cpu = model.cpu()
        model_cpu.eval()
        example = torch.randint(0, len(VOCAB), (1, MAX_LEN), dtype=torch.long)
        traced = torch.jit.trace(model_cpu, example)
        traced.save(scripted_path)
        print("Saved TorchScript to", scripted_path)
    except Exception as e:
        print("TorchScript export провалился:", e)
    try:
        qmodel = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
        qpath = os.path.join("artifacts", f"{prefix}_quantized.pt")
        torch.save(qmodel.state_dict(), qpath)
        print("Сохраненный квантованный указатель состояния модели в", qpath)
    except Exception as e:
        print("Сбой квантования:", e)


def load_artifacts(prefix: str = CHAR_CNN_MODEL_PREFIX) -> nn.Module:
    state_path = os.path.join(ARTIFACTS_DIR, f"{prefix}.pt")
    vocab_path = os.path.join(ARTIFACTS_DIR, f"{prefix}_vocab.json")
    if not os.path.exists(state_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError("Artifacts не найдены. Сначала тренируйте!")
    with open(vocab_path, "r", encoding="utf-8") as f:
        _vocab = json.load(f)
    model = CharCNN(vocab_size=len(_vocab))
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_prob(model: nn.Module, text: str) -> float:
    model.eval()
    seq = encode_text(text)
    xb = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        prob = torch.sigmoid(logits).item()
    return prob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--samples", type=int, default=200000, help="Количество примеров для генерации (default 200000)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument(
        "--out-prefix", type=str, default=CHAR_CNN_MODEL_PREFIX, help="Prefix для сохраненных artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    print("Конфиг:", args)
    X, y = generate_dataset(total_samples=args.samples)
    # split
    split = int(0.9 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print("Сгенерированный датасет:", len(X_train), "train /", len(X_val), "val")

    train_ds = PhoneDataset(X_train, y_train)
    val_ds = PhoneDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    model = CharCNN(vocab_size=len(VOCAB))
    model, history = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    metrics = evaluate(model, val_loader)
    save_artifacts(model, prefix=args.out_prefix)

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", f"{args.out_prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)
    print("Saved metrics.")

    tests = [
        "+7 (912) 345-67-89",
        "hello world",
        "привет мир",
        "12345",
        "+380671234567",
        "ID-ABC12345",
        "+44 20 7946 0958",
        "+1-800-FLOWERS",
        "+1-800-ЦВЕТЫ"
        "tel:+49 (151) 12345678 ext 12",
        "Order#1234567890",
        "Заказ#1234567890",
        "user-abc-123456",
        "Тел: +7 925 123 45 67",
        "Tel: +7 925 123 45 67",
        "Phone: +7 925 123 45 67",
        "+86 137 1234 5678",
        "+1 (202) 555-0182",
        "000000000000000000",
    ]
    print("Ручные тесты:")
    for t in tests:
        p = predict_prob(model, t)
        print(f"{t!r} -> {p:.4f} -> {'PHONE' if p>=0.5 else 'NOT'}")


if __name__ == "__main__":
    main()