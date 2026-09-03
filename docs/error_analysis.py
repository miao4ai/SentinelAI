# %% [markdown]
# # 融合错误分析 — 各位置判错了哪些样本
#
# 在真实 XD-Violence **3 模态**数据（视觉 I3D + 音频 AST + 文本 ASR）上，用**电影分组
# 交叉验证**为每个片段拿到 **out-of-fold（OOF）预测**，然后把**被判错的样本**挑出来看：
#
# - 每个融合位置（单模态 / ①②③④⑤⑥）各自错在哪、错多少
# - **所有方法都判错**的"最难"样本（真·模糊）
# - **① early vs ⑤ late 的分歧样本**（深融合帮到 / 帮倒忙的地方）
# - **跨模态冲突**样本（模态互相矛盾）——各融合怎么裁决
#
# > 片段名本身就是线索：`电影名.年份__#起止时间_label_X`（`label_A`=正常，其余=暴力）。
# > 前置：在 GPU 机器上跑，`data/xd-violence/` 下要有 i3d_rgb / audio_full / text_features。

# %%
import glob, os, logging, warnings
from collections import defaultdict
import numpy as np, pandas as pd, torch
# 让 Lightning 别把 GPU/TPU 横幅刷进每个 cell 的输出
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from sentinelai.coordinated_fusion import CoordinatedFusion
from sentinelai.early_fusion import JointFusionTransformer
from sentinelai.train.lit_module import LitCrossAttention
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V_TOK, A_TOK = 32, 16
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 46)

def b(n): return n.replace(".npy", "").replace(".mp4", "")
def is_violent(n): return 0 if "_label_A" in n else 1
def resample(x, n):
    if len(x) == 0: return np.zeros((n, x.shape[1]), np.float32)
    return x[np.linspace(0, len(x) - 1, n).round().astype(int)].astype(np.float32)

# %% [markdown]
# ## 1. 载入三模态特征（保留每片段的 key，用于回看错例）

# %%
def load_i3d():
    out = {}
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            a = np.load(f); a = a.mean(1) if a.ndim == 3 else a
            out[b(os.path.basename(f))] = a.astype(np.float32)
    return out
def load_npz(folder):
    out = {}
    for f in glob.glob(f"{folder}/*.npz"):
        z = np.load(f, allow_pickle=True); out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out

vis, aud, txt = load_i3d(), load_npz(f"{DATA}/audio_full"), load_npz(f"{DATA}/text_features")
keys = sorted(k for k in vis if k in aud and k in txt)
y = np.array([is_violent(k) for k in keys])
groups = np.array([k.split("__")[0] for k in keys])
movie = [k.split("__")[0] for k in keys]
print(f"{len(keys)} 个 3 模态片段, {len(set(groups))} 部电影, {y.mean():.0%} 暴力")

Vp = np.stack([vis[k].mean(0) for k in keys])
Ap = np.stack([aud[k] for k in keys])
Tp = np.stack([txt[k] for k in keys])
Vs = np.stack([resample(vis[k], V_TOK) for k in keys])
allp = np.concatenate([Vp, Ap, Tp], 1)
cv = list(GroupKFold(5).split(np.arange(len(keys)), y, groups))

# %% [markdown]
# ## 2. 每个位置的 out-of-fold 预测
# 单模态 + ③拼接 + ④决策树 + ⑤平均 用 sklearn 的 `cross_val_predict`（同一 CV）；
# ①early / ②coordinated / ⑥cross-attn 是 torch/lightning，手动在每折训练、对测试折出概率。

# %%
def sk_oof(X):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

pv, pa, pt = sk_oof(Vp), sk_oof(Ap), sk_oof(Tp)          # 单模态
p3 = sk_oof(allp)                                        # ③ 拼接
p5 = (pv + pa + pt) / 3                                  # ⑤ 平均

# ④ 决策级：GBDT stack 各模态概率（用内层 OOF 概率训练，防泄漏）
p4 = np.zeros(len(keys))
for tr, te in cv:
    g = groups[tr]
    def inner(X):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        return cross_val_predict(pipe, X[tr], y[tr], cv=list(GroupKFold(3).split(X[tr], y[tr], g)), method="predict_proba")[:, 1]
    meta = GradientBoostingClassifier(random_state=0).fit(np.c_[inner(Vp), inner(Ap), inner(Tp)], y[tr])
    p4[te] = meta.predict_proba(np.c_[pv[te], pa[te], pt[te]])[:, 1]

# %%
def torch_oof_early_coord(kind):
    """① early 或 ② coordinated 的 OOF 概率。"""
    Ts = Tp[:, None, :]; As = Ap[:, None, :]           # 文本/音频各作 1 token
    out = np.zeros(len(keys)); dims = {"visual": 2048, "audio": 768, "text": 768}
    for tr, te in cv:
        torch.manual_seed(0)
        model = (JointFusionTransformer(dims, d_model=128, n_layers=2, n_categories=1) if kind == "early"
                 else CoordinatedFusion(dims, d_model=128, n_categories=1)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
        if kind == "early":
            tr_in = {"visual": Vs[tr], "audio": As[tr], "text": Ts[tr]}; te_in = {"visual": Vs[te], "audio": As[te], "text": Ts[te]}
        else:
            tr_in = {"visual": Vp[tr], "audio": Ap[tr], "text": Tp[tr]}; te_in = {"visual": Vp[te], "audio": Ap[te], "text": Tp[te]}
        yt = torch.tensor(y[tr], dtype=torch.float32, device=DEVICE)[:, None]
        T = {m: torch.tensor(v, device=DEVICE) for m, v in tr_in.items()}
        E = {m: torch.tensor(v, device=DEVICE) for m, v in te_in.items()}
        for _ in range(200):
            model.train(); opt.zero_grad(); o = model(T); lo = o[0] if isinstance(o, tuple) else o
            lf(lo, yt).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            o = model(E); lo = o[0] if isinstance(o, tuple) else o
            out[te] = torch.sigmoid(lo).squeeze(1).cpu().numpy()
    return out

p1 = torch_oof_early_coord("early")
p2 = torch_oof_early_coord("coord")

# %%
def crossattn_oof():
    """⑥ cross-attn：I3D 帧作 K/V，[音频,文本] 作 Query。"""
    G = np.stack([Ap, Tp], 1)                          # (N,2,768) 两个 query token
    out = np.zeros(len(keys))
    for tr, te in cv:
        pl.seed_everything(0, verbose=False)
        lit = LitCrossAttention(video_dim=2048, guide_dim=768, n_categories=1, d_model=128, n_heads=4, lr=1e-3)
        ds = TensorDataset(torch.tensor(Vs[tr]), torch.tensor(G[tr]), torch.tensor(y[tr][:, None], dtype=torch.float32))
        pl.Trainer(max_epochs=50, accelerator="auto", devices=1, logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                   num_sanity_val_steps=0).fit(lit, DataLoader(ds, batch_size=128, shuffle=True))
        lit.eval()
        with torch.no_grad():
            out[te] = lit.model.predict_proba(torch.tensor(Vs[te], device=lit.device),
                                              torch.tensor(G[te], device=lit.device))[:, 0].cpu().numpy()
    return out

p6 = crossattn_oof()

# %% [markdown]
# ## 3. 汇总成一张表，算各方法的错误数

# %%
probs = {"visual": pv, "audio": pa, "text": pt, "①early": p1, "②coord": p2,
         "③concat": p3, "④gbdt": p4, "⑤late": p5, "⑥xattn": p6}
df = pd.DataFrame({"key": keys, "movie": movie, "label": y})
for m, p in probs.items():
    df[m] = p; df[m + "_pred"] = (p >= 0.5).astype(int)

rows = []
for m, p in probs.items():
    pred = (p >= 0.5).astype(int)
    rows.append({"方法": m, "F1": f1_score(y, pred), "P": precision_score(y, pred),
                 "R": recall_score(y, pred), "错误数": int((pred != y).sum()),
                 "漏报(FN)": int(((pred == 0) & (y == 1)).sum()),
                 "误报(FP)": int(((pred == 1) & (y == 0)).sum())})
summary = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
summary.round(3)

# %% [markdown]
# ## 3.5 误分类明细表（长表：一条错例一行）
# 每个**被判错的样本**一行，带上：**方法（融合种类）**、**错误类型**（FN 漏报 / FP 误报）、
# 样本本身（key、电影）、以及各模态/该方法的概率。可按 `方法` 或 `错误类型` 任意筛选来分析。

# %%
records = []
for m, p in probs.items():
    pred = (p >= 0.5).astype(int)
    for i in np.where(pred != y)[0]:
        records.append({
            "方法": m,
            "错误类型": "FN漏报" if y[i] == 1 else "FP误报",
            "key": keys[i], "电影": movie[i],
            "真实": int(y[i]), "预测": int(pred[i]), "该方法概率": round(float(p[i]), 2),
            "视觉": round(float(pv[i]), 2), "音频": round(float(pa[i]), 2), "文本": round(float(pt[i]), 2),
        })
mis = pd.DataFrame(records)
print(f"总误分类记录：{len(mis)} 条（跨 {len(probs)} 种方法、{mis['key'].nunique()} 个不同片段）")
# 每种方法的 FN / FP 分布
mis.groupby(["方法", "错误类型"]).size().unstack(fill_value=0)

# %% [markdown]
# 明细样本（默认按 电影+片段 排序，方便看同一片段被哪些方法、以什么类型判错）。
# 换筛选看具体情况，例如：
# `mis[mis["方法"] == "⑤late"]`　或　`mis[mis["错误类型"] == "FP误报"]`　或　`mis[mis["电影"].str.contains("...")]`

# %%
mis.sort_values(["电影", "key", "方法"]).head(40)

# %% [markdown]
# ## 4. 所有方法都判错的"最难"样本
# 每个片段被多少种方法判错；被 9 种全判错的，是真·模糊/标注可疑的样本。

# %%
pred_cols = [m + "_pred" for m in probs]
df["错误方法数"] = sum((df[c] != df["label"]) for c in pred_cols)
hard = df[df["错误方法数"] == len(probs)][["key", "label", "错误方法数"] + list(probs)]
print(f"被全部 {len(probs)} 种方法判错的样本：{len(hard)} 个")
hard.round(2)

# %% [markdown]
# ## 5. ① early vs ⑤ late 的分歧样本
# 交叉点实验里 ① early 数据够时反超 ⑤ late。看具体是哪些片段让它们分道扬镳。

# %%
ea = (df["①early_pred"] == df["label"]) & (df["⑤late_pred"] != df["label"])
le = (df["⑤late_pred"] == df["label"]) & (df["①early_pred"] != df["label"])
print(f"① 对而 ⑤ 错：{ea.sum()} 个　|　⑤ 对而 ① 错：{le.sum()} 个")
print("\n— ① early 救回来的（⑤ 错、① 对）—")
display_cols = ["key", "label", "visual", "audio", "text", "①early", "⑤late"]
df[ea][display_cols].round(2).head(12)

# %%
print("— ⑤ late 更稳的（① 错、⑤ 对）—")
df[le][display_cols].round(2).head(12)

# %% [markdown]
# ## 6. 跨模态冲突样本 —— 模态互相矛盾时各融合怎么裁决
# 挑出**单模态判断互相打架**的片段（有的模态说暴力、有的说正常），看各融合的最终裁决。
# 文本最弱（对物理暴力几乎无信号），这里最容易看到"融合要不要信文本"。

# %%
single = df[["visual", "audio", "text"]].values
disagree = (single >= 0.5).sum(1)                      # 0..3 个模态投暴力
df["模态分歧"] = np.minimum(disagree, 3 - disagree)   # 1 = 最分歧(2:1)，0 = 一致
conflict = df[df["模态分歧"] == 1].copy()
print(f"三模态出现 2:1 分歧的片段：{len(conflict)} 个")
cols = ["key", "label", "visual", "audio", "text", "②coord", "④gbdt", "⑤late", "⑥xattn"]
conflict.sort_values("text")[cols].round(2).head(15)

# %% [markdown]
# ## 7. 文本把 ⑤ 平均带偏的例子
# 文本是最弱模态。挑"视觉+音频都对、但文本强烈反向"的片段，看 naive 平均 ⑤ 是否被文本拖错、
# 而学习式融合（②/④/⑥）有没有顶住。

# %%
va_right = ((df["visual"] >= 0.5) == df["label"]) & ((df["audio"] >= 0.5) == df["label"])
text_wrong = (df["text"] >= 0.5) != df["label"]
misled = df[va_right & text_wrong]
print(f"视觉+音频对、文本反向的片段：{len(misled)} 个")
print(f"其中 ⑤late 被带错：{int(((misled['⑤late']>=0.5)!=misled['label']).sum())} 个"
      f"　②coord 被带错：{int(((misled['②coord']>=0.5)!=misled['label']).sum())} 个"
      f"　④gbdt 被带错：{int(((misled['④gbdt']>=0.5)!=misled['label']).sum())} 个")
misled[["key", "label", "visual", "audio", "text", "②coord", "④gbdt", "⑤late", "⑥xattn"]].round(2).head(12)

# %% [markdown]
# ## 8. 错例可视化 —— 拉原始数据（帧 / 音频 / 转写）来看
# 给定一个片段 key，把**原始画面、原始音频（波形+频谱+可播放）、原始转写文本**都拉出来，
# 配上各方法对它的预测，方便直接肉眼分析"到底错在哪"。转写已缓存（抽文本时存了 `text`），
# 帧和音频从 HF 原视频现抽。

# %%
import tempfile, subprocess
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio, display

HF_REPO = "jherng/xd-violence"
# 原始转写文本（抽文本时已缓存在 text_features 的 npz 的 "text" 字段里）
txt_raw = {}
for f in glob.glob(f"{DATA}/text_features/*.npz"):
    z = np.load(f, allow_pickle=True)
    txt_raw[str(z["key"])] = (str(z["text"]) if "text" in z.files else "")

_vmap = {}
def vmap():
    global _vmap
    if not _vmap:
        from huggingface_hub import list_repo_files
        _vmap = {b(os.path.basename(f)): f for f in list_repo_files(HF_REPO, repo_type="dataset") if f.endswith(".mp4")}
    return _vmap

def decode_audio(path, sr=16000):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-f", "s16le", "-ac", "1",
                          "-ar", str(sr), "-loglevel", "error", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0, sr

def inspect_clip(key, n_frames=8):
    """错例可视化：原始帧 + 音频(波形/频谱/播放器) + 转写 + 各方法预测。"""
    from huggingface_hub import hf_hub_download
    row = df[df.key == key].iloc[0]
    print(f"clip: {key}\n电影: {row.movie}  |  真实标签: {'暴力' if row.label else '正常'}")
    tbl = [{"方法": m, "概率": round(float(row[m]), 2),
            "预测": "暴力" if row[m + "_pred"] else "正常",
            "对错": "✗错" if row[m + "_pred"] != row.label else "✓对"} for m in probs]
    display(pd.DataFrame(tbl))
    print(f"📝 原始转写: {txt_raw.get(key) or '(无对白 / 零向量)'}")
    vm = vmap()
    if key not in vm:
        print("（HF 上没有此片段的原视频，跳过帧/音频）"); return
    with tempfile.TemporaryDirectory() as tmp:
        vp = hf_hub_download(HF_REPO, vm[key], repo_type="dataset", local_dir=tmp)
        try:
            dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nk=1:nw=1", vp]).strip() or 0) or 10.0
        except Exception:
            dur = 10.0
        # 原始帧
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", vp, "-vf",
                        f"fps={max(n_frames/dur,0.001)}", "-frames:v", str(n_frames), f"{tmp}/f_%02d.jpg"])
        fs = sorted(glob.glob(f"{tmp}/f_*.jpg"))
        if fs:
            fig, ax = plt.subplots(1, len(fs), figsize=(2.2 * len(fs), 2.4))
            for a, fp in zip(np.atleast_1d(ax), fs):
                a.imshow(Image.open(fp)); a.axis("off")
            fig.suptitle("original frames", fontsize=9); plt.show()
        # 原始音频：波形 + 频谱 + 可播放
        wav, sr = decode_audio(vp)
        if len(wav):
            fig, ax = plt.subplots(2, 1, figsize=(10, 4))
            ax[0].plot(np.linspace(0, len(wav) / sr, len(wav)), wav, lw=0.4)
            ax[0].set(title="waveform", xlabel="sec")
            ax[1].specgram(wav, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
            ax[1].set(title="spectrogram", xlabel="sec", ylabel="Hz")
            plt.tight_layout(); plt.show()
            display(Audio(wav, rate=sr))

# %% [markdown]
# **示例**：挑一个被**所有方法都判错**的最难样本，把原始数据拉出来看它为什么难。
# 换任意 key 即可分析别的错例，例如 `inspect_clip(mis[mis["方法"]=="⑤late"].iloc[0]["key"])`。

# %%
demo_key = df[df["错误方法数"] == len(probs)].iloc[0]["key"] if (df["错误方法数"] == len(probs)).any() else keys[0]
inspect_clip(demo_key)

# %% [markdown]
# ## 结论（看数字填）
# - 各方法错误数见 §3；全判错的"硬样本"见 §4（多半是镜头模糊/标注边界）。
# - §5 显示 ① early 和 ⑤ late 各自救回的片段，解释交叉点为什么发生在样本层面。
# - §6/§7 显示**弱文本冲突**时，naive 平均 ⑤ 容易被带偏，而能学权重的 ②/④/⑥ 更稳——
#   这正是"学习式融合对模态质量差异鲁棒"在**单个错例**上的体现。
