# Experiment: CLIP image retrieval (B) vs caption retrieval (A)

**Question:** to make document figures searchable, is it better to embed the
figure's *image* with a CLIP visual encoder (architecture B), or to embed the
*M3-generated caption text* with our text model (architecture A)?

**Setup:** same 98 (deduplicated) rendered figures from the `rag_lab_docs`
corpus, indexed both ways (brute-force cosine):
- **A — caption**: M3 caption → MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`, 384-d)
- **B — CLIP**: image → `clip-ViT-B-32` (512-d); query → `clip-ViT-B-32-multilingual-v1`

Eval: 6 queries each targeting one paper's framework figure. hit@5 = a figure
from the right paper appears in the top-5.

| query | A caption | B CLIP |
|---|:---:|:---:|
| GNN 四种架构示意图 | ✓ | ✓ |
| skeleton 时空图结构 | ✓ | ✓ |
| MaPLe 框架图 | ✓ | ✗ |
| CLIP2Scene 3D点云 | ✓ | ✗ |
| Vita-CLIP 结构 | ✓ | ✗ |
| SpeechMedAssist 框架 | ✓ | ✗ |

**Result: A = 6/6 (1.00), B = 2/6 (0.33).**

## Why
- CLIP won only the two *visually-distinctive* diagram queries (multiple
  architecture sketches; skeleton spatio-temporal graph).
- CLIP lost the four queries that name a method (MaPLe / CLIP2Scene / Vita-CLIP).
  Those names are **text printed inside the figure** — the M3 caption reads them,
  but a CLIP visual embedding cannot read in-figure text.

## Honest caveat
The eval queries mostly *name* the method (a textual cue), which inherently
favours A. A purely visual query ("a diagram with three boxes joined by arrows")
would favour B. So the takeaway is not "CLIP is useless":

> For document / framework figures whose meaning lives in **text and structure**,
> caption-based retrieval wins decisively. CLIP-style image embeddings suit
> **natural-image / appearance** search — which this corpus is not.

In the production pipeline we keep A for retrieval and additionally feed the real
image to M3 at answer time (best of both). CLIP could be added as an extra recall
channel for appearance-style queries if the corpus grew to include photos.

Reproduce: `python -m rag_lab.clip_index --config configs/docs.yaml --build`
then `--compare` (or `--query "..."` for a side-by-side).
