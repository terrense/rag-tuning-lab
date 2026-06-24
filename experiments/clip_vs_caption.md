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

## Follow-up: visual queries + A+B fusion

To be fair to CLIP we added a second eval set of *appearance-only* queries (no
method names) and an A+B RRF-fused method:

| eval set | A caption | B CLIP | A+B fused |
|---|:---:|:---:|:---:|
| method-named (6 q) | 6/6 | 2/6 | 6/6 |
| visual-appearance (4 q) | 4/4 | 1/4 | 4/4 |

**CLIP loses even the visual queries** ("human skeleton keypoints", "3D point
cloud scene"). Two reasons, and the first is the important one:

1. **We feed CLIP whole-page renders, not cropped figures.** To avoid the
   image-XObject explosion we render the entire page; a page is mostly text, so
   CLIP's image embedding describes "an academic-paper page", not "a skeleton
   diagram" — its visual advantage is diluted away. CLIP needs cropped figure
   regions to shine.
2. The multilingual CLIP text tower is a distilled model, weaker than the native
   English CLIP text encoder.

**A+B fusion = A here** (no gain, because A already saturates the eval) **but
never worse than A** — RRF means an extra weak channel can't drop A's hits. So
adding CLIP as an extra recall channel is *safe* but unhelpful on this corpus.

## Takeaways
> For document / framework figures whose meaning lives in **text and structure**,
> M3-caption retrieval wins decisively — even on visually-phrased queries.
> CLIP would need **cropped figures** (region detection) and/or an
> **appearance-heavy corpus** (photos) to contribute. Fusion is harmless.

Pipeline keeps A for retrieval and feeds the real image to M3 at answer time
(best of both). Future work to actually help CLIP: detect+crop figure regions
before embedding.

Reproduce: `python -m rag_lab.clip_index --config configs/docs.yaml --build`
then `--compare` (or `--query "..."` for a side-by-side).
