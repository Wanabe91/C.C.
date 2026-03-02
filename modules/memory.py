import json, os, logging
from datetime import datetime

logger = logging.getLogger("memory")

DEFAULT_PROFILE = {
    "name": "",
    "preferences": {},   # "кофе": "американо", "язык": "русский"
    "beliefs": [],        # ["ценит приватность", "любит минимализм"]
    "goals": [],          # ["запустить PersonalAI"]
    "relationships": {},  # "жена": "Марина"
    "notes": [],          # произвольные заметки
}


# ═══════════════════════════════════════════════
# СЛОЙ 1 — Рабочая память (RAM, живёт до перезапуска)
# ═══════════════════════════════════════════════
class WorkingMemory:
    def __init__(self, limit=20):
        self.limit = limit
        self._h = []

    def add(self, role, content):
        self._h.append({"role": role, "content": content})
        if len(self._h) > self.limit:
            self._h = self._h[-self.limit:]

    def get(self): return list(self._h)
    def clear(self): self._h.clear()
    def __len__(self): return len(self._h)


# ═══════════════════════════════════════════════
# СЛОЙ 2 — Эпизодическая память (ChromaDB)
# ═══════════════════════════════════════════════
class EpisodicMemory:
    def __init__(self, db_path="data/episodic", embed_model=None):
        os.makedirs(db_path, exist_ok=True)
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self._col = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        ).get_or_create_collection("episodes", metadata={"hnsw:space": "cosine"})

        model = embed_model or \
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._enc = SentenceTransformer(model)
        logger.info(f"EpisodicMemory: {self._col.count()} записей")

    def _embed(self, text):
        return self._enc.encode(text, normalize_embeddings=True).tolist()

    def save(self, content, role="dialog", tags=None):
        ts  = datetime.utcnow().isoformat()
        uid = f"{ts}_{abs(hash(content)) % 100000}"
        self._col.add(
            ids=[uid],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[{"role": role, "ts": ts, "tags": json.dumps(tags or [])}],
        )

    def recall(self, query, top_k=5, role_filter=None):
        n = min(top_k, max(self._col.count(), 1))
        kwargs = dict(
            query_embeddings=[self._embed(query)],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if role_filter:
            kwargs["where"] = {"role": role_filter}
        r = self._col.query(**kwargs)
        return [
            {"content": doc,
             "role":    m.get("role", ""),
             "ts":      m.get("ts", "")[:10],
             "dist":    round(d, 3)}
            for doc, m, d in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]

    def count(self): return self._col.count()


class NullEpisodicMemory:
    def save(self, content, role="dialog", tags=None):
        return None

    def recall(self, query, top_k=5, role_filter=None):
        return []

    def count(self):
        return 0


# ═══════════════════════════════════════════════
# СЛОЙ 3 — Профиль личности (JSON, всегда в промпте)
# ═══════════════════════════════════════════════
class PersonalityProfile:
    def __init__(self, path="data/profile.json"):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._d = json.load(f)
        else:
            self._d = dict(DEFAULT_PROFILE)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._d, f, ensure_ascii=False, indent=2)

    def get(self): return dict(self._d)

    def to_prompt(self) -> str:
        """Текстовый блок для system prompt."""
        d = self._d
        lines = ["## Профиль пользователя"]
        if d.get("name"):
            lines.append(f"Имя: {d['name']}")
        if d.get("preferences"):
            lines.append("Предпочтения: " +
                ", ".join(f"{k}: {v}" for k, v in d["preferences"].items()))
        if d.get("beliefs"):
            lines.append(f"Убеждения: {', '.join(d['beliefs'])}")
        if d.get("goals"):
            lines.append(f"Цели: {', '.join(d['goals'])}")
        if d.get("relationships"):
            lines.append("Важные люди: " +
                ", ".join(f"{k} — {v}" for k, v in d["relationships"].items()))
        if d.get("notes"):
            lines.append(f"Заметки: {'; '.join(d['notes'])}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def update(self, updates: dict):
        """LLM передаёт dict с изменениями — умный merge."""
        for k, v in updates.items():
            if k not in self._d:
                continue
            cur = self._d[k]
            if isinstance(cur, dict) and isinstance(v, dict):
                cur.update(v)
            elif isinstance(cur, list) and isinstance(v, list):
                for item in v:
                    if item not in cur:
                        cur.append(item)
            else:
                self._d[k] = v
        self._save()


# ═══════════════════════════════════════════════
# ФАСАД — Planner работает только с этим классом
# ═══════════════════════════════════════════════
class Memory:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        episodic_path = cfg.get("episodic_db") or cfg.get("long_term_db") or "data/episodic"
        if episodic_path.endswith(".db"):
            episodic_path = os.path.join(os.path.dirname(episodic_path) or "data", "episodic")

        self.working  = WorkingMemory(limit=cfg.get("short_term_limit", 20))
        if cfg.get("episodic_enabled", True):
            try:
                self.episodic = EpisodicMemory(
                    db_path=episodic_path,
                    embed_model=cfg.get("embed_model"),
                )
            except Exception as exc:
                logger.warning("Episodic memory disabled: %s", exc)
                self.episodic = NullEpisodicMemory()
        else:
            self.episodic = NullEpisodicMemory()
        self.profile  = PersonalityProfile(
            path=cfg.get("profile_path", "data/profile.json")
        )

    # ── Рабочая ──────────────────────────────
    def add(self, role, content): self.working.add(role, content)
    def get_history(self): return self.working.get()

    # ── Эпизодическая ────────────────────────
    def save(self, content, role="dialog", tags=None):
        self.episodic.save(content, role=role, tags=tags)

    def recall(self, query, top_k=5):
        return self.episodic.recall(query, top_k=top_k)

    def recall_as_text(self, query, top_k=5) -> str:
        mems = self.episodic.recall(query, top_k=top_k)
        if not mems: return ""
        return "## Из долгосрочной памяти\n" + \
               "\n".join(f"[{m['ts']}] {m['content']}" for m in mems)

    # ── Профиль ──────────────────────────────
    def get_profile_prompt(self): return self.profile.to_prompt()
    def update_profile(self, updates): self.profile.update(updates)

    # ── Вызывать после каждого обмена ────────
    def after_turn(self, user: str, assistant: str):
        self.episodic.save(
            f"Пользователь: {user}\nАссистент: {assistant}",
            role="dialog"
        )

    def stats(self):
        return {
            "рабочая_память": len(self.working),
            "эпизодов":       self.episodic.count(),
            "имя":            self.profile.get().get("name", "не задано"),
        }
