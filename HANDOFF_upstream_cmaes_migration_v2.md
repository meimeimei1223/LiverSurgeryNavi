# 引継ぎ書 v2 — CMA-ES を本家 Hansen c-cmaes に置換(並列の決定性をタダで得る)

最終更新: 2026-06-02 / 対象: LiverSurgeryNavi(`AAA_LiverSurgeryNaviComb` / GitHub `meimeimei1223/LiverSurgeryNavi`)
作業言語: 日本語。識別子・コードは英語。**この文書は次チャットがそのまま着手できる完全仕様**です。
**本 v2 は v1 を置き換えます**(レビュー反映版)。

> ⚠️ **最重要の前提**: この変更は Ctrl+G / Ctrl+Shift+G / Ctrl+I など **BIPOP 系の全エンジンに影響**します。壊すと術中ナビの根幹が止まります。**フォールバック付き・段階移行・各段で検証** を厳守(本書 §4・§7・§8)。

### v1 → v2 変更サマリ(本家ソース突き合わせレビューの反映)
- **【FIX①】Test の static バッファ競合**: 本家 `cmaes_TestForTermination(t)` は停止理由を関数内 `static char sTestOutString[3024]` に sprintf して返す。**数値(best x/RMSE)は per-instance で決定的だが、返り文字列バッファは全インスタンス共有** → 並列で stop_reason が化ける/データレース(UB)。アダプタの Test ラッパーを `#pragma omp critical` で囲み、evo 所有バッファへ strncpy して返す。
- **【FIX②】再シードは `cmaes_random_Start`(`cmaes_random_init` ではない)**: `_init` は毎回 `rgrand`(256B)を再確保しリークする。`Start` は再確保せず `flgstored=0`・`aktseed=seed` をセット、seed≥1 で乱数列は完全同一。
- **【FIX③】改名+`#include "cmaes.c"` は `.c` シムに閉じる**: `.cpp` に入れると ANSI C を C++ コンパイルになり `void*` 暗黙変換で詰まり得る。改名は C の小シムに閉じ、アダプタ(`.cpp`)は `extern "C"` で受ける。
- **【追加】他の共有 static の棚卸し**(`getTimeStr`/`SayHello`/`WriteToFile`/`ReadSignals`/`ERRORMESSAGE`)と**並列領域内で呼ばない**指針。
- **【追加】`pop_bounded` は per-instance(evo 所有)必須**(static 化禁止)。
- **【確認済】§8 の「`evo->` 全参照 grep」を実施**: 5 エンジンすべて **`gen`/`lambda`/`sigma` のみ**(付録 A)。アダプタの 3 フィールドミラー設計は全エンジンで成立。

---

## 0. 現状(安全に戻した状態 = 移行のベースライン)

`/mnt/user-data/outputs/` に以下、**これが移行の出発点**:
- `CmaesRefineV3R.h` … 実行ループ並列化の機構は実装済み、**既定 OFF**(`inline bool g_v3rParallelRuns = false;` @ 988)。
- `CmaesRefineV3RS.h` … 同上、**既定 OFF**(`inline bool g_v3rsParallelRuns = false;` @ 416)。
- `main.cpp` … 並列トグルのチェックボックス実装済み(`SilCompare::drawSection()` の直後 ~5261、V3RS のグローバルにバインドし V3R へミラー)。

並列機構は正しく動き高速化も確認済み(Ctrl+G ~200ms→~45ms、Ctrl+Shift+G/I ~1300ms→~650ms)。OFF にしてあるのは決定性の問題のためで、本移行で根治する。

---

## 1. なぜ本家へ置換するのか(背景と確定診断)

### 1-1. 現 `third_party/c-cmaes/cmaes.c` は **Hansen 本家ではない**(別物の再実装)
ヘッダは Hansen / c-cmaes URL を名乗るが中身は「チュートリアル arXiv:1604.00772 を元にゼロから書いた ~500 行の単一ファイル」。本家(~3117 行, version "3.20.00.beta")とは別物。確定根拠:

| 項目 | 現 cmaes.c(再実装) | 本家 Hansen c-cmaes |
|---|---|---|
| `cmaes_init` 署名 | `cmaes_t* cmaes_init(int N, const double* xstart, double sigma0, int lambda, const double* lb, const double* ub)`(戻り値確保, **6 引数**) | `cmaes_init(cmaes_t* t, int dim, double* xstart, double* stddev, long inseed, int lambda, const char* filename)`(第一引数 t, **inseed/filename あり, 7 引数**) |
| 乱数 | **グローバル `rand()`** + `randn()` 内の関数内 `static int have; static double spare;` | **per-instance** `cmaes_random_t`(`t->rand`: `flgstored`/`hold`, `cmaes_random_Gauss(&t->rand)`)。グローバル rand 無し |
| 固有値分解 | Jacobi(`eigen_sym`) | Householder2 + QLalgo2 |
| 終了判定 | `cmaes_TestForTermination(evo, maxgen, tolfun, tolx)`(**4 引数**) | `cmaes_TestForTermination(t)`(**1 引数**, 基準は `t->sp.stop*`) |
| 箱拘束 | `cmaes_init(lb,ub)` で**内部クランプ** | コアは**クランプしない**。`boundary_transformation.c` で変換 |
| 戦略定数 | 2016 チュートリアル系(`cc=(4+mueff/N)/…`, `c1=2/((N+1.3)²+mueff)` 等) | 旧式(`cc≈4/(N+4)`, `cs=(mueff+2)/(N+mueff+3)`, `ccov/mucov` 系) |

`CmaesUtils.h:1364` が `cmaes_init(DIM, xstart, params.sigma0, params.lambda, lb, ub)`(6 引数)= ビルドに入っているのは**再実装版で確定**。

### 1-2. 並列が直列と一致しない根本原因(確定)
再実装 `randn()` は **グローバル `rand()` + 関数内 static `have`/`spare`**(Box–Muller のペア繰り越し)。直列は 10 run が `have`/`spare` を**逐次引き継ぐ**(run 間結合)。並列はその順序を再現できず**分岐**。A/B ログで **Run1(index0)だけ一致・他は stop 理由も gen 数も相違**という繰り越し由来の指紋を確認済み。`cmaes_tls_wrapper.c` は `rand()`→`rand_r` で一様乱数のみ thread-local 化するが、**`have`/`spare` は randn() 内ブロックスコープ static でラッパーの file スコープ宣言に隠れて効かない**(ISO C 6.2.1)。さらに per-run `srand(cma_seed)` は**別 TU** でマクロ非適用。⇒ **ラッパーだけでは不可能**。

### 1-3. 本家なら「並列の決定性が(数値については)タダ」 — ただし共有 static は別途処理
本家は乱数が **per-instance**(`t->rand`)。各 run が自前の `cmaes_init(..., inseed=cma_seed, ...)` で**自分の種だけ**で初期化されるので run 間結合が**構造的に存在しない**。各スレッドが自分の `cmaes_t` だけ触るので **best x / RMSE などの数値は決定的・レースなし**。

> **【重要・v2 訂正】** 「設計上スレッド安全・レースなし」は**数値については真**だが、**関数内 static バッファについては成り立たない**。特に `cmaes_TestForTermination` の戻り文字列(`static char sTestOutString[3024]`)は全インスタンス共有で、並列だとデータレース(UB)+ stop_reason 化け。→ §3・§4-1【FIX①】で `critical` 化。`inseed=cma_seed` を渡せば数値は **直列==並列・実行ごと再現・機種/コンパイラ非依存**(glibc `rand()` は環境依存だが本家の専用 RNG はどこでも同一)。

### 1-4. 代償(合意済み)
**新ベースライン**。RNG・固有値分解・箱拘束・**戦略定数**(§1-1 表)がすべて変わるため、**run0 を含め全 run の結果が旧と変わり、過去保存姿勢(PoseLibrary)は再現しない**。精度は本家のほうが枯れており同等以上が期待できるが**フル再検証が必要**(§7)。論文 methods には**どちらの定数系か(式)を明記**しておくこと。

> 表記注意: README/ヘッダが「c-cmaes Apache-2.0」を名乗るが実体は別物。本家置換で表記が正しくなる(ライセンス Apache-2.0/LGPL-2.1 は現状と互換)。

---

## 2. 目標
1. `third_party/c-cmaes` を**本家 Hansen 実体**に置換(+ `boundary_transformation`)。
2. 各 run を `inseed=cma_seed` 相当で再シード → **数値が並列=直列・完全決定的**。
3. **BIPOP 系全エンジン(V1/V2/V3/V3R/V3RS)を壊さない**。
4. **即時フォールバック可能**(`USE_UPSTREAM_CMAES` で旧 cmaes.c へ 1 フラグ復帰)。
5. 既存の高速化機構(実行ループ並列・top-K・UI トグル)は**そのまま活かす**。

---

## 3. 本家 API の要点(次チャットが必要な事実)

取得元: `github.com/CMA-ES/c-cmaes/src/` の **5 ファイルのみ** — `cmaes.c` / `cmaes.h` / `cmaes_interface.h` / `boundary_transformation.c` / `boundary_transformation.h`(`example_*.c` や `plotcmaesdat.*`、`*.par` は**不要**。`cmaes_init_para(filename="non")` でファイル IO を止めるので `*.par` は置かない方が安全)。

- 初期化(分割版が安全): `cmaes_init_para(&t, dim, xstart, stddev, inseed, lambda, filename)` → 必要なら `t.sp.stop*` を設定 → `cmaes_init_final(&t)`。
  - `stddev` は**座標ごとの初期標準偏差配列**。スカラ sigma0 を使うなら **全要素 = sigma0**(本家は `sigma=sqrt(Σstddev²/N)` なので全要素 sigma0 で `sigma=sigma0` に一致)。
  - `inseed`: **0=clock(非決定)**。**非ゼロ必須・2e9 未満**(cma_seed ~2,026万は OK)。アダプタでは init 時に placeholder `inseed=1` を渡し、後述 `set_seed` で本シードに差し替える。
  - `filename`: **`"non"`**(または `"no"`/`"none"`)を渡してファイル読み書きを止める。
- **【FIX②】再シード(run ごと)は `cmaes_random_Start(&t.rand, cma_seed)`**:
  - `cmaes_random_init(&t.rand, seed)` は**毎回 `t.rgrand=new_void(32,…)`(256B)を再確保**し、`init_final` が既に 1 回確保済み → run ごと `_init` で**リーク**(`cmaes_exit` は最後の 1 個のみ free)。
  - `cmaes_random_Start` は**再確保せず**既存 `rgrand` を使い `flgstored=0`・`aktseed=seed` をセット。seed≥1 のとき `_init` も内部で `Start` を呼ぶだけ → **乱数列は完全同一・リークだけ消える**。
- 反復: `pop = cmaes_SamplePopulation(&t)`(**unbounded** な `double*const*`、内部バッファ→free しない)→ fitness 計算 → `cmaes_UpdateDistribution(&t, fitvals)`。
  - **【追加】** `cmaes_UpdateDistribution` は等価適応度時に `ERRORMESSAGE("Warning: sigma increased…")` を stderr へ出す(毎世代経路)。sigma 調整は per-instance で**数値は決定的**、ただし stderr 出力が並列で交錯する(クラッシュ/UB ではなく見栄えのみ)。気になるなら黙らせる(優先度低)。
- 終了: `const char* s = cmaes_TestForTermination(&t)`(NULL なら継続)。制御は `t.sp.stopMaxIter`(最大世代)/`t.sp.stopTolFun`/`t.sp.stopTolX`/`t.sp.stopTolFunHist` 等。**正確なフィールド名は実 `cmaes.h`(`cmaes_readpara_t`)で確認**。
  - **【FIX①】** 返り値は**関数内 `static char sTestOutString[3024]` へのポインタ**(全インスタンス共有)。並列では sprintf がレース。停止判定・数値は per-instance なので**数値は無事**だが、**返り文字列は §4-1 のラッパーで `critical` 化して evo 所有バッファへコピー**。
  - **本家は既定で多数の基準が有効**(TolFunHist/ConditionCov/NoEffectAxis/NoEffectCoord/EqualFunVals 等)→ stop 理由・gen 数は再実装版と**変わる**(想定内。BIPOP 側は文字列/gen 数を読むだけで動作は壊れない)。再実装版に寄せたいなら余分な基準を 0/無効に。
- 取得: `cmaes_GetPtr(&t, "xbestever")`(best-ever, **unbounded**)、`cmaes_Get(&t,"fbestever"/"sigma"/"generation"/"eval")`。**現エンジンはどれも未使用**(best は pop+fval から自前抽出)。
- 箱拘束(別モジュール): `cmaes_boundary_transformation_init(&b, lb, ub, dim)` / `cmaes_boundary_transformation(&b, x_unbounded, x_bounded, dim)`(変換) / `..._inverse(...)` / `..._exit(&b)`。**評価は変換後 x で行い、Update には内部の unbounded pop が使われる**(`example_boundary.c` の流儀)。
- **【追加】他の共有 static の棚卸し**: `getTimeStr()` の `static char s[33]`、`cmaes_SayHello`、`cmaes_ReSampleSingle`/`FATAL`/`ERRORMESSAGE` の static。いずれも**並列領域内で呼ばなければ安全**。`cmaes_SayHello`/`cmaes_WriteToFile*`/`cmaes_ReadSignals` を run ループ内で**呼ばない**(初期化ログは並列前に 1 回)。`filename="non"` なので signals 読みは元々走らない。

---

## 4. 推奨アーキテクチャ — **アダプタ shim + フォールバックフラグ**(リスク最小)

全エンジンが**完全同一の 6 引数 API**を使い、読む構造体は **`gen`/`lambda`/`sigma` の 3 つだけ**(付録 A で全エンジン確認済)。best は pop+fval から各エンジンが**自前抽出**(`evo->xbest` 不使用)。よって**旧 API を被せたアダプタ 1 枚**でエンジンをほぼ無改変にできる。

### 4-1. アダプタが公開するもの(= 現エンジンが呼ぶ名前と同一)

公開型(ON ビルドの `cmaes.h`): エンジンが触る 3 フィールド + 隠し実体ポインタ。
```c
/* cmaes.h  —  USE_UPSTREAM_CMAES のとき */
typedef struct cmaes_s {
    int    gen;     /* HANSEN gen をミラー    */
    int    lambda;  /* init で設定             */
    double sigma;   /* HANSEN sigma をミラー   */
    void*  impl;    /* AdapterImpl*（隠し）   */
} cmaes_t;

cmaes_t*    cmaes_init(int N, const double* xstart, double sigma0, int lambda,
                       const double* lb, const double* ub);
void        cmaes_set_seed(cmaes_t*, unsigned int seed);          /* 新規: srand 置換 */
double**    cmaes_SamplePopulation(cmaes_t*);
void        cmaes_UpdateDistribution(cmaes_t*, const double* fval);
const char* cmaes_TestForTermination(cmaes_t*, int maxgen, double tolfun, double tolx);
void        cmaes_exit(cmaes_t*);
```

隠し実体(`cmaes_adapter.cpp`):
```cpp
struct AdapterImpl {
    HANSEN_cmaes_t                   h;             /* by value, per-instance     */
    cmaes_boundary_transformation_t  b;             /* per-instance               */
    double**                         pop_bounded;   /* lambda×N, per-instance【static禁止】*/
    int                              N, lambda;
    char                             stop_buf[4096]; /* >= 3024, per-instance      */
};
```

実装(要点):
- `cmaes_init(N,xstart,sigma0,lambda,lb,ub)`
  - `evo` と `AdapterImpl` を確保。`stddev[N]=sigma0`(全要素)。
  - `HANSEN_cmaes_init_para(&im->h, N, xstart, stddev, /*inseed*/1, lambda, "non")` → `HANSEN_cmaes_init_final(&im->h)`(ここで `rgrand` を 1 回確保)。
  - `cmaes_boundary_transformation_init(&im->b, lb, ub, N)`。`pop_bounded` を lambda×N で確保。
  - `evo->lambda=lambda; evo->sigma=im->h.sigma; evo->gen=0; evo->impl=im;`
- **【FIX②】** `cmaes_set_seed(evo, seed)`(各エンジンの `srand(cma_seed)` を置換)
  ```cpp
  AdapterImpl* im = (AdapterImpl*)evo->impl;
  HANSEN_cmaes_random_Start(&im->h.rand, (long unsigned)(seed ? seed : 1u)); /* 再確保しない */
  ```
- `cmaes_SamplePopulation(evo)`
  - `pu = HANSEN_cmaes_SamplePopulation(&im->h);` 各 k で `cmaes_boundary_transformation(&im->b, pu[k], im->pop_bounded[k], N);`
  - `evo->gen=im->h.gen; evo->sigma=im->h.sigma;` `return im->pop_bounded;`(**エンジンは bounded x を見る=従来同等**)
- `cmaes_UpdateDistribution(evo, fval)`(fval は bounded で評価済み、Update は内部 unbounded arx を使う=正しい)
  - `HANSEN_cmaes_UpdateDistribution(&im->h, fval);` `evo->gen=im->h.gen; evo->sigma=im->h.sigma;`
- **【FIX①】** `cmaes_TestForTermination(evo, maxgen, tolfun, tolx)`
  ```cpp
  AdapterImpl* im = (AdapterImpl*)evo->impl;
  im->h.sp.stopMaxIter = maxgen;          /* フィールド名は cmaes.h で確認 */
  im->h.sp.stopTolFun  = tolfun;
  im->h.sp.stopTolX    = tolx;
  const char* s;
  #pragma omp critical (cmaes_test)        /* 共有 static バッファを保護 */
  {
      s = HANSEN_cmaes_TestForTermination(&im->h);  /* 判定は per-instance、文字列は共有 static */
      if (s) { std::strncpy(im->stop_buf, s, sizeof(im->stop_buf)-1);
               im->stop_buf[sizeof(im->stop_buf)-1] = '\0'; }
      else   { im->stop_buf[0] = '\0'; }
  }
  return im->stop_buf[0] ? im->stop_buf : NULL;   /* evo 所有バッファを返す */
  ```
  Test は 1 世代 1 回・極小なので `critical` の実害なし。判定・数値は変わらない。
- `cmaes_exit(evo)`
  - `cmaes_boundary_transformation_exit(&im->b); HANSEN_cmaes_exit(&im->h);`(`rgrand` を 1 回 free)。`pop_bounded`/`im`/`evo` 解放。

> **正しさの根拠**: 評価は bounded pop で行い、Update は本家内部 unbounded `arx`+fval で進む(`example_boundary.c` と同型で正しい)。エンジンが記録する best も bounded x →**そのまま実姿勢として使える**。`xbestever`(unbounded)はどのエンジンも読まないので変換不要。

### 4-2. 【FIX③】シンボル衝突の回避 — 改名は **`.c` シム**に閉じる
`.cpp` に `#include "cmaes.c"` を入れると 2900 行の ANSI C を C++ コンパイルになり `void*` 暗黙変換で詰まり得る。**本家は C のまま**にして、改名だけ小 `.c` シムに閉じ、アダプタ(`.cpp`)は `extern "C"` で受ける。

`hansen_renames.h`(改名マクロ、共有):
```c
#define cmaes_t                   HANSEN_cmaes_t
#define cmaes_random_t            HANSEN_cmaes_random_t
#define cmaes_readpara_t          HANSEN_cmaes_readpara_t
#define cmaes_timings_t           HANSEN_cmaes_timings_t
#define cmaes_init                HANSEN_cmaes_init
#define cmaes_init_para           HANSEN_cmaes_init_para
#define cmaes_init_final          HANSEN_cmaes_init_final
#define cmaes_SamplePopulation    HANSEN_cmaes_SamplePopulation
#define cmaes_UpdateDistribution  HANSEN_cmaes_UpdateDistribution
#define cmaes_TestForTermination  HANSEN_cmaes_TestForTermination
#define cmaes_GetPtr              HANSEN_cmaes_GetPtr
#define cmaes_Get                 HANSEN_cmaes_Get
#define cmaes_exit                HANSEN_cmaes_exit
#define cmaes_random_init         HANSEN_cmaes_random_init
#define cmaes_random_Start        HANSEN_cmaes_random_Start
#define cmaes_random_Gauss        HANSEN_cmaes_random_Gauss
#define cmaes_random_Uniform      HANSEN_cmaes_random_Uniform
#define cmaes_random_exit         HANSEN_cmaes_random_exit
/* … cmaes_interface.h の全公開シンボルを機械生成で網羅
   (cmaes_readpara_*, cmaes_timings_*, cmaes_resume_distribution,
    cmaes_WriteToFile*, cmaes_ReadFromFilePtr, cmaes_ReadSignals,
    cmaes_SayHello, cmaes_PerturbSolutionInto, cmaes_ReSampleSingle*,
    cmaes_SetMean, cmaes_Optimize, cmaes_FATAL …)。型の改名も忘れない。
   cmaes_Get と cmaes_GetPtr は別トークンなので各々必要。
   ※ boundary_transformation_* は改名不要(現コードに同名なし)。*/
```

`hansen_cmaes_renamed.c`(**C としてコンパイル**):
```c
#include "hansen_renames.h"
#include "hansen/cmaes.c"     /* 本家実体。cmaes_interface.h → cmaes.h を引く */
```

`cmaes_adapter.cpp`(**C++**):
```cpp
extern "C" {
  #include "hansen_renames.h"            /* 改名を当てた状態で… */
  #include "hansen/cmaes_interface.h"    /* …宣言と HANSEN_cmaes_t 構造体を取り込む */
  #include "hansen/boundary_transformation.h"
}
#include "cmaes.h"                        /* エンジン公開型(cmaes_t)＋6引数API宣言 */
/* §4-1 の cmaes_init/set_seed/Sample/Update/Test/exit をここで実装 */
```
構造体定義(`cmaes.h`)は C++ でも問題なく通る。**詰まるのは実装本体だけ**なので `.c` 側に隔離される。**改名漏れ=未定義/重複は最大の地雷**(§8)。`cmaes_interface.h` の宣言から機械生成を徹底。

### 4-3. フォールバック(術中ナビ必須)
CMake オプション **`USE_UPSTREAM_CMAES`(初期値 OFF)**:
- OFF → 現 `cmaes.c` をコンパイル(**現状の挙動を完全維持**)。`cmaes_set_seed` は参照されない(エンジンの差し替えは `#ifdef` 済、§5-2)。
- ON → `cmaes_adapter.cpp` + `hansen_cmaes_renamed.c` + `boundary_transformation.c` をコンパイル。`cmaes.h` は ON 用の最小公開型に切替(§4-1)。`cmaes_tls_wrapper.c` と旧 `cmaes.c` は除外。
- **回帰が出たら OFF に戻すだけで即復帰**。各エンジン検証が済むまで ON 既定にしない。`-fopenmp` は維持。

> `cmaes_tls_wrapper.c` は本家経路では**不要**(per-instance RNG が自前でスレッド安全。残る共有 static は §3/§4-1①②③で処理)。ON 時はビルドから外す。

---

## 5. 触る/足すファイル一覧

### 5-1. 追加
- 本家実体(`third_party/c-cmaes/hansen/`): `cmaes.c`, `cmaes.h`, `cmaes_interface.h`, `boundary_transformation.c`, `boundary_transformation.h`(github master の 5 ファイル。コミットハッシュを記録)。
- `third_party/c-cmaes/hansen_renames.h`(改名マクロ、§4-2)。
- `third_party/c-cmaes/hansen_cmaes_renamed.c`(C シム、§4-2)。
- `third_party/c-cmaes/cmaes_adapter.cpp`(C++ アダプタ、§4-1/4-2)。

### 5-2. 変更
- `CMakeLists.txt`: `USE_UPSTREAM_CMAES` 分岐。ON で adapter+renamed.c+boundary をコンパイル、`cmaes_tls_wrapper.c`/旧 `cmaes.c` を除外。`-fopenmp` 維持。
- `cmaes.h`: `#ifdef USE_UPSTREAM_CMAES` で ON 用の最小公開型(§4-1)+ `cmaes_set_seed` 宣言。`#else` は現行の型・宣言を維持。
- **各エンジンの per-run シード 1 行を差し替え**(`#ifdef` で両経路維持):
  ```cpp
  if (rc.cma_seed != 0) {        /* V1 は params.rng_seed */
  #ifdef USE_UPSTREAM_CMAES
      cmaes_set_seed(evo, (unsigned int)rc.cma_seed);
  #else
      srand(rc.cma_seed);
  #endif
      /* 既存の "[Vxx] Deterministic seed: ..." ログはそのまま */
  }
  ```
  行アンカー(現状ファイル基準):

| ファイル | `cmaes_init` | `srand`→`cmaes_set_seed` | Sample | Update | Test | exit |
|---|---|---|---|---|---|---|
| `CmaesUtils.h`(V1) | 1364 | **1378** | 1516 | 1589 | 1616 | 1622 |
| `CmaesRefineV2.h` | 969 | **972** | 1001 | 1024 | 1044 | 1053 |
| `CmaesRefineV3.h` | 1515 | **1518** | 1563 | 1593 | 1614 | 1622 |
| `CmaesRefineV3R.h` | 2183 | **2186** | 2217 | 2269 | 2289 | 2297 |
| `CmaesRefineV3RS.h` | 3006 | **3009** | 3117 | 3226 | 3246 | 3254 |

> `cmaes_init`/`Sample`/`Update`/`Test`/`exit` の**呼び出しは変更不要**(アダプタが同一署名)。`evo->gen/lambda/sigma` 参照も変更不要(アダプタがミラー)。`cmaes_TestForTermination` の 4 引数署名も**そのまま**(アダプタが受けて本家へ橋渡し+`critical`)。

### 5-3. 退役
- `third_party/c-cmaes/cmaes_tls_wrapper.c`(ON 経路では使わない)。

---

## 6. 不変で残すもの(壊さない)
- **実行ループ並列機構**(V3R/V3RS): per-run の jitter/sigma0 を `rng`(mt19937)から **8 draw 順で事前生成** → `exec_run` ラムダ → `g_v3r(s)ParallelRuns` で parallel/serial 分岐 → last-run 読み出し。**cmaes 内部に触れず run_one_bipop を呼ぶだけ**なので本家化で**真に決定的**になる。維持。
- V3R の**後段 top-K**(kTopK=3 で full-RMSE 再評価)と best-run 選択(@2751 付近)。
- V3RS の**内側ラスタ OMP**(projection ~867, splat ~999)の `&& !omp_in_parallel()` ガード。
- main.cpp の**並列トグル UI**(両エンジンへミラー)。
- 無関係機能(RMSE-cap トグル、Alt+P 捕捉、F9 比較、Checkpoint)。

---

## 7. 段階移行手順 + 検証プロトコル(各段で必ず)

**順序(低リスク順):**
1. 本家 5 ファイル投入 + `hansen_renames.h` + `hansen_cmaes_renamed.c` + `cmaes_adapter.cpp` + `cmaes.h` の ON 型/`set_seed` + `CMakeLists` 分岐。**まず ON でコンパイルを通す**(エンジンは 5 行のシード差し替えのみ)。改名漏れ=リンクエラーを潰す。
2. **Ctrl+G(V3R)を最初に通す**。同一姿勢で:
   - **(a) 直列(トグル OFF)**で実行 → 収束が健全か(IoU/RMSE が現行と同程度。**数値は変わってよい**が品質が落ちていないこと)。
   - **(b) 直列==並列がビット一致**(数値=核心)。**【FIX①適用後は stop_reason も一致するはず**(未適用だと文字列だけ化ける)。
   - **(c) 並列を同一姿勢で 2 回** → **2 回が完全一致**(実行ごと再現)。
   - stop 理由/gen 数/total_gens が健全か(理由は変わるが破綻していないこと)。
3. 次に **Ctrl+Shift+G / Ctrl+I(V3RS)** → 同じ (a)(b)(c)。pure-IoU 経路(`g_ctrlgsPureIoUMode`)も確認。
4. 次に **Shift+G(V3)**、**Alt+Shift+G(V2)**、**Alt+G(V1)** → コンパイル維持 & 直列健全性。
5. 全エンジンの (a)〜(c) が OK なら `USE_UPSTREAM_CMAES` を ON 既定 → 並列トグルを段階的に ON 既定へ。

**回帰時:** `USE_UPSTREAM_CMAES=OFF` で即復帰 → 切り分け後に再挑戦。
**再調整:** 定数系/RNG/境界/停止が変わるため sigma0・lambda・停止しきい値の再調整が要る場合あり。`min_match_ratio`(0.30)や bad-eval ゲート(`matched<matched_min → penalty_value=9.9`)はエンジン側ロジックで不変。**収束の質を必ず目視+数値で確認**。

---

## 8. 落とし穴チェックリスト
- [ ] **【FIX②】** 再シードは `cmaes_random_Start`(`_init` は `rgrand` リーク)。`set_seed` は Start を呼ぶ。
- [ ] **【FIX①】** `cmaes_TestForTermination` ラッパーを `#pragma omp critical` で囲み、戻り文字列を evo 所有 `stop_buf` へ strncpy して返す(共有 `static char sTestOutString[3024]` のレース回避、UB)。
- [ ] **【FIX③】** 改名+`#include "cmaes.c"` は **C シム**に閉じ、アダプタは `extern "C"`(ANSI C を C++ コンパイルしない)。型(`cmaes_t`/`cmaes_random_t`/`cmaes_readpara_t`/`cmaes_timings_t`)の改名も忘れない。`cmaes_Get`/`cmaes_GetPtr` は各々必要。改名は `cmaes_interface.h` から機械生成。
- [ ] **【追加】** `pop_bounded` は **per-instance(evo 所有)**。static 禁止。
- [ ] **【追加】** `cmaes_SayHello`/`cmaes_WriteToFile*`/`cmaes_ReadSignals` を **並列領域内で呼ばない**(共有 static)。flat-fitness の `ERRORMESSAGE`(stderr)は数値に無害・交錯のみ(任意で黙らせる)。
- [ ] `inseed` は **非ゼロ・2e9 未満**。アダプタ init は placeholder=1 → `set_seed` で本シード。
- [ ] `cmaes_init_para` の `filename` は **`"non"`**。`*.par` は置かない。
- [ ] スカラ sigma0 → **stddev 全要素 = sigma0**。
- [ ] **箱拘束は変換方式**(評価は bounded x、Update は内部 unbounded)。クランプではない。
- [ ] `t.sp.stopMaxIter`/`stopTolFun`/`stopTolX` の**正確なフィールド名を実 `cmaes.h` で確認**。本家既定の余分な停止基準で stop 理由・gen 数が変わる(正常)。寄せたいなら無効化。
- [ ] **per-run lambda**(BIPOP global は lambda 倍化)を `cmaes_init` にそのまま渡す。
- [ ] **【確認済】** `evo->` 参照は全エンジン `gen`/`lambda`/`sigma` のみ(付録 A)。今後エンジンを足すときも `C`/`D`/`ps`/`xbest` を直接読まないこと(読むならアダプタ拡張)。
- [ ] **境界 xstart**: 既定 7-DoF は `xstart={0,…,0}`=区間 `[-1,1]` の内側なので変換方式でも退化しない。**境界ちょうどから始めるモードがあれば**そこだけ内側に寄せる。
- [ ] ライセンス/表記(README・ヘッダ)を本物の c-cmaes に更新。

---

## 9. 次チャットへの最初の指示(コピペ用)

> 「LiverSurgeryNavi の CMA-ES を本家 Hansen c-cmaes に置換する。方針は HANDOFF **v2** の §4(アダプタ shim + `USE_UPSTREAM_CMAES` フォールバック)で、レビュー反映の **3 修正は必須**:①`cmaes_TestForTermination` を `#pragma omp critical`+evo 所有 `stop_buf` でラップ(共有 static バッファのレース回避)、②再シードは `cmaes_random_Start`(`_init` は rgrand リーク)、③改名+`#include "cmaes.c"` は C シムに閉じてアダプタは `extern "C"`。本家 5 ファイルを `third_party/c-cmaes/hansen/` に置く前提で、`hansen_renames.h`/`hansen_cmaes_renamed.c`(C)/`cmaes_adapter.cpp`(C++、§4-1 の 6 引数 API+`cmaes_set_seed`+箱拘束変換+gen/lambda/sigma ミラー+`pop_bounded` は per-instance)を作り、`cmaes.h` に ON 用最小型+`cmaes_set_seed`、`CMakeLists.txt` に `USE_UPSTREAM_CMAES` 分岐。各エンジンは §5-2 の表の `srand`→`cmaes_set_seed` 1 行を `#ifdef` で差し替え。検証は §7 の (a) 直列健全 → (b) 直列==並列ビット一致(①適用後 stop_reason も一致)→ (c) 並列 2 回一致 を Ctrl+G(V3R)から順に。`/mnt/project` は読み取り専用なので `/home/claude` で編集し `/mnt/user-data/outputs` に出す。」
>
> 添付してほしいもの: 本家 `cmaes.h` と `cmaes_interface.h`(改名リスト確定 + `sp.stop*` フィールド名確認用)、`boundary_transformation.h`、現行 `cmaes.h`(現 `cmaes_t` 型)。

---

## 付録 A. 棚卸し結果(現状ファイル基準)

**cmaes API 呼び出し(全署名同一):**
```
CmaesUtils.h(V1):     init@1364  srand@1378  Sample@1516  Update@1589  Test@1616  exit@1622
CmaesRefineV2.h:      init@969   srand@972   Sample@1001  Update@1024  Test@1044  exit@1053
CmaesRefineV3.h:      init@1515  srand@1518  Sample@1563  Update@1593  Test@1614  exit@1622
CmaesRefineV3R.h:     init@2183  srand@2186  Sample@2217  Update@2269  Test@2289  exit@2297
CmaesRefineV3RS.h:    init@3006  srand@3009  Sample@3117  Update@3226  Test@3246  exit@3254
RegistrationActions.h: p.rng_seed=cma_base+run @4374(V1 経路、srand はエンジン側)
```
署名: `cmaes_init(DIM,xstart,sigma0,lambda,lb,ub)` / `cmaes_SamplePopulation(evo)` / `cmaes_UpdateDistribution(evo,fval.data())` / `cmaes_TestForTermination(evo,maxgen,tolfun,tolx)` / `cmaes_exit(evo)`。

**`evo->` 直接参照(grep 実測, 全エンジン)= `gen`/`lambda`/`sigma` のみ:**
```
CmaesUtils.h:   gen×1 lambda×2 sigma×1
CmaesRefineV2.h:  gen×1 lambda×3 sigma×1
CmaesRefineV3.h:  gen×1 lambda×3 sigma×1
CmaesRefineV3R.h: gen×1 lambda×3 sigma×1
CmaesRefineV3RS.h:gen×1 lambda×3 sigma×1
RegistrationActions.h: (なし)
```
→ アダプタは Sample/Update のたびに `gen`/`sigma` をミラー、`lambda` は init で設定。best は pop+fval からエンジン自前抽出(`xbestever` 変換不要)。

## 付録 B. ディスパッチ(参考, main.cpp)
Ctrl+G → V3R(`runBipopCmaesV3R`)、Ctrl+Shift+G → V3RS(`runBipopCmaesV3RS`)、Ctrl+I → V3I→`g_ctrlgsPureIoUMode=true`→V3RS、Shift+G → V3、Alt+G → V1、Alt+Shift+G → V2。dispatch ~1665–1922。

## 付録 C. レビュー 3 修正の要約(クイックリファレンス)
| # | 症状 | 原因 | 修正 |
|---|---|---|---|
| ① | 並列で stop_reason が化ける/UB(数値は無事) | `cmaes_TestForTermination` が関数内 `static char sTestOutString[3024]` を共有 | Test ラッパーを `#pragma omp critical`+evo 所有 `stop_buf` へ strncpy |
| ② | 登録を繰り返すと 256B ずつリーク | `cmaes_random_init` が毎回 `rgrand` 再確保 | `set_seed` で `cmaes_random_Start` を使う(乱数列同一・確保なし) |
| ③ | `.cpp` で C を C++ コンパイル → 詰まり得る | `#include "cmaes.c"` を `.cpp` に置く | 改名を C シムに閉じ、アダプタは `extern "C"` |
