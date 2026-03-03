"""
LLM Memory Viewer
Запуск: streamlit run memory_viewer.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import json
import os
import numpy as np

st.set_page_config(
    page_title="LLM Memory Viewer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
.stApp { background: #0d0d14; }

.stat-card {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a45;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.stat-card .num { font-size: 2.2rem; font-weight: 800; color: #7b7bff; line-height: 1; }
.stat-card .label { font-size: 0.78rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: .08em; }

.doc-card {
    background: #13131f;
    border: 1px solid #222235;
    border-left: 3px solid #5b5bff;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.doc-id { font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #5b5bff; margin-bottom: 6px; }
.doc-text { color: #c8c8e8; font-size: 0.9rem; line-height: 1.55; }
.doc-meta { font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #555; margin-top: 8px; }
.vec-badge {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3a3a6a;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono';
    font-size: 0.7rem;
    color: #9999ff;
    margin-top: 6px;
}
.search-hit { border-left-color: #22c55e !important; }
.search-score {
    display: inline-block;
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono';
    font-size: 0.7rem;
    color: #4ade80;
    margin-left: 8px;
}
section[data-testid="stSidebar"] { background: #0a0a11 !important; border-right: 1px solid #1e1e35; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #13131f !important;
    border: 1px solid #2a2a45 !important;
    color: #c8c8e8 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono' !important;
}
.stSelectbox > div > div {
    background: #13131f !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3a3aff, #7b7bff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
}
.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-weight: 700; }
.stTabs [aria-selected="true"] { color: #7b7bff !important; }
div[data-testid="stExpander"] { background: #13131f; border: 1px solid #222235 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 8px 0 24px;">
    <div style="font-family:'Syne';font-size:2rem;font-weight:800;color:#7b7bff;letter-spacing:-.02em;">
        🧠 LLM Memory Viewer
    </div>
    <div style="color:#555;font-size:.85rem;margin-top:2px;">Просмотр памяти локальной LLM — SQLite + ChromaDB</div>
</div>
""", unsafe_allow_html=True)

tab_sqlite, tab_chroma = st.tabs(["🗃️  SQLite", "🔮  ChromaDB"])


# ══════════════════════════════════════════════════════════════════════════════
# SQLite
# ══════════════════════════════════════════════════════════════════════════════
with tab_sqlite:
    with st.sidebar:
        st.markdown("### 🗃️ SQLite")
        db_path = st.text_input("Путь к .db файлу", placeholder="/path/to/memory.db", key="sqlite_path")

    if not db_path:
        st.info("👈 Укажи путь к SQLite файлу в боковой панели", icon="💡")
    elif not os.path.exists(db_path):
        st.error(f"Файл не найден: `{db_path}`")
    else:
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cursor.fetchall()]

            if not tables:
                st.warning("База данных пуста — таблиц не найдено.")
            else:
                file_size_kb = os.path.getsize(db_path) / 1024
                total_rows = 0
                for t in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM [{t}]")
                        total_rows += cursor.fetchone()[0]
                    except:
                        pass

                cols_stat = st.columns(3)
                for col, (num, label) in zip(cols_stat, [
                    (len(tables), "таблиц"),
                    (total_rows, "строк всего"),
                    (f"{file_size_kb:.1f} KB", "размер файла"),
                ]):
                    col.markdown(f'<div class="stat-card"><div class="num">{num}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_left, col_right = st.columns([1, 3])

                with col_left:
                    selected_table = st.selectbox("Таблица", tables)
                    cursor.execute(f"PRAGMA table_info({selected_table})")
                    col_info = cursor.fetchall()
                    col_names = [c[1] for c in col_info]
                    col_types = [c[2] for c in col_info]
                    cursor.execute(f"SELECT COUNT(*) FROM [{selected_table}]")
                    row_count = cursor.fetchone()[0]
                    st.markdown(f"**Строк:** {row_count}")
                    st.markdown("**Колонки:**")
                    for name, dtype in zip(col_names, col_types):
                        st.markdown(f"`{name}` <span style='color:#555;font-size:.75rem'>{dtype}</span>", unsafe_allow_html=True)

                with col_right:
                    search_term = st.text_input("🔎 Поиск по содержимому (LIKE)", placeholder="Введи текст...", key="sqlite_search")
                    limit = st.slider("Строк", 10, min(500, max(10, row_count)), 50, key="sqlite_limit")

                    if search_term:
                        text_cols = [n for n, t in zip(col_names, col_types)
                                     if "TEXT" in t.upper() or "CHAR" in t.upper() or t == ""]
                        if text_cols:
                            like_clauses = " OR ".join([f"[{c}] LIKE ?" for c in text_cols])
                            params = [f"%{search_term}%"] * len(text_cols)
                            df = pd.read_sql_query(
                                f"SELECT * FROM [{selected_table}] WHERE {like_clauses} LIMIT {limit}",
                                conn, params=params)
                            st.caption(f"Найдено: {len(df)} строк по «{search_term}»")
                        else:
                            df = pd.read_sql_query(f"SELECT * FROM [{selected_table}] LIMIT {limit}", conn)
                            st.warning("Текстовых колонок не найдено")
                    else:
                        df = pd.read_sql_query(f"SELECT * FROM [{selected_table}] LIMIT {limit}", conn)

                    st.dataframe(df, use_container_width=True, height=380)
                    st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{selected_table}.csv", mime="text/csv")

                st.divider()
                st.markdown("#### ✏️ Произвольный SQL запрос")
                custom_sql = st.text_area("SQL", value=f"SELECT * FROM [{tables[0]}] LIMIT 20", height=80)
                if st.button("▶️ Выполнить"):
                    try:
                        result_df = pd.read_sql_query(custom_sql, conn)
                        st.dataframe(result_df, use_container_width=True)
                        st.success(f"✅ Строк: {len(result_df)}")
                    except Exception as e:
                        st.error(f"SQL ошибка: {e}")
        except Exception as e:
            st.error(f"Ошибка открытия БД: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB
# ══════════════════════════════════════════════════════════════════════════════
with tab_chroma:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔮 ChromaDB")
        chroma_path = st.text_input("persist_directory", placeholder="/path/to/chroma_db", key="chroma_path")

    if not chroma_path:
        st.info("👈 Укажи путь к папке ChromaDB в боковой панели", icon="💡")
    elif not os.path.exists(chroma_path):
        st.error(f"Папка не найдена: `{chroma_path}`")
    else:
        try:
            import chromadb

            @st.cache_resource
            def get_chroma(path):
                return chromadb.PersistentClient(path=path)

            client = get_chroma(chroma_path)
            collections = client.list_collections()

            if not collections:
                st.warning("Коллекций не найдено.")
            else:
                total_docs = sum(c.count() for c in collections)
                cols_stat = st.columns(2)
                for col, (num, label) in zip(cols_stat, [
                    (len(collections), "коллекций"),
                    (total_docs, "документов"),
                ]):
                    col.markdown(f'<div class="stat-card"><div class="num">{num}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                selected_col_name = st.selectbox("Коллекция", [c.name for c in collections])
                collection = client.get_collection(selected_col_name)
                doc_count = collection.count()

                col_main, col_viz = st.columns([2, 1])

                with col_main:
                    # Семантический поиск
                    st.markdown("#### 🔍 Семантический поиск")
                    search_query = st.text_input("Запрос", placeholder="Введи текст для поиска по смыслу...", key="chroma_search")
                    n_results = st.slider("Результатов", 1, min(20, doc_count), 5)

                    if search_query:
                        try:
                            res = collection.query(
                                query_texts=[search_query],
                                n_results=n_results,
                                include=["documents", "metadatas", "distances", "embeddings"],
                            )
                            embs_q = res.get("embeddings") or [[None] * n_results]
                            for doc, meta, dist, emb in zip(
                                res["documents"][0], res["metadatas"][0],
                                res["distances"][0], embs_q[0]
                            ):
                                score = max(0, 1 - dist)
                                vec_info = f"<span class='vec-badge'>📐 {len(emb)}D вектор</span>" if emb else ""
                                meta_str = json.dumps(meta, ensure_ascii=False) if meta else ""
                                st.markdown(f"""
                                <div class="doc-card search-hit">
                                    <span class="search-score">score {score:.3f}</span>
                                    <div class="doc-text" style="margin-top:8px">{doc or '<em>пусто</em>'}</div>
                                    {vec_info}
                                    <div class="doc-meta">{meta_str}</div>
                                </div>""", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Ошибка поиска: {e}")

                    st.markdown("#### 📄 Все документы")
                    text_filter = st.text_input("Фильтр по тексту", placeholder="Часть текста...", key="chroma_filter")
                    limit = st.slider("Документов", 5, min(100, doc_count), 20, key="chroma_limit")

                    try:
                        raw = collection.get(
                            limit=limit,
                            include=["documents", "metadatas", "embeddings"],
                            where_document={"$contains": text_filter} if text_filter else None,
                        )
                    except Exception:
                        raw = collection.get(limit=limit, include=["documents", "metadatas", "embeddings"])

                    ids = raw.get("ids", [])
                    docs = raw.get("documents", [])
                    metas = raw.get("metadatas", [])
                    embeddings = raw.get("embeddings") or []

                    st.caption(f"Показано: {len(ids)} документов")

                    for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
                        emb = embeddings[i] if embeddings and i < len(embeddings) else None
                        vec_info = f"<span class='vec-badge'>📐 {len(emb)}D  [{emb[0]:.3f}, {emb[1]:.3f}, …]</span>" if emb is not None else ""
                        meta_str = json.dumps(meta, ensure_ascii=False) if meta else ""
                        st.markdown(f"""
                        <div class="doc-card">
                            <div class="doc-id">{doc_id}</div>
                            <div class="doc-text">{doc or '<em>пусто</em>'}</div>
                            {vec_info}
                            <div class="doc-meta">{meta_str}</div>
                        </div>""", unsafe_allow_html=True)

                    export = [{"id": i, "document": d, "metadata": m} for i, d, m in zip(ids, docs, metas)]
                    st.download_button(
                        "⬇️ Экспорт JSON",
                        json.dumps(export, ensure_ascii=False, indent=2),
                        file_name=f"{selected_col_name}.json",
                        mime="application/json",
                    )

                with col_viz:
                    st.markdown("#### 📊 Визуализация эмбеддингов")

                    if not embeddings or len([e for e in embeddings if e is not None]) < 3:
                        st.caption("Нужно минимум 3 документа с векторами")
                    else:
                        try:
                            from sklearn.decomposition import PCA
                            import plotly.express as px

                            valid = [(e, d) for e, d in zip(embeddings, docs) if e is not None]
                            emb_matrix = np.array([v[0] for v in valid], dtype=float)
                            labels = [
                                (v[1][:45] + "…" if v[1] and len(v[1]) > 45 else v[1] or "—")
                                for v in valid
                            ]

                            if emb_matrix.shape[0] >= 3 and emb_matrix.shape[1] >= 2:
                                pca = PCA(n_components=2)
                                reduced = pca.fit_transform(emb_matrix)

                                fig = px.scatter(
                                    x=reduced[:, 0], y=reduced[:, 1],
                                    hover_name=labels,
                                    color_discrete_sequence=["#7b7bff"],
                                    title="PCA 2D проекция",
                                )
                                fig.update_layout(
                                    paper_bgcolor="#13131f",
                                    plot_bgcolor="#0d0d14",
                                    font=dict(color="#c8c8e8", family="JetBrains Mono", size=11),
                                    title_font_color="#7b7bff",
                                    xaxis=dict(gridcolor="#1e1e35", zerolinecolor="#2a2a45", title="PC1"),
                                    yaxis=dict(gridcolor="#1e1e35", zerolinecolor="#2a2a45", title="PC2"),
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    height=340,
                                )
                                fig.update_traces(marker=dict(size=10, opacity=0.85))
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(
                                    f"Дисперсия: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
                                    f"PC2 {pca.explained_variance_ratio_[1]:.1%}"
                                )
                            else:
                                st.caption("Недостаточно данных")
                        except ImportError:
                            st.warning("`pip install scikit-learn plotly`")
                        except Exception as e:
                            st.error(f"Ошибка визуализации: {e}")

                    st.markdown("#### ℹ️ Инфо")
                    st.markdown(f"""
                    <div class="doc-card" style="border-left-color:#f59e0b">
                        <div class="doc-id">COLLECTION</div>
                        <div class="doc-text">
                            Имя: <b>{selected_col_name}</b><br>
                            Документов: <b>{doc_count}</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    try:
                        meta_info = collection.metadata or {}
                        if meta_info:
                            st.json(meta_info)
                    except:
                        pass

        except ImportError:
            st.error("ChromaDB не установлен → `pip install chromadb`")
        except Exception as e:
            st.error(f"Ошибка ChromaDB: {e}")
