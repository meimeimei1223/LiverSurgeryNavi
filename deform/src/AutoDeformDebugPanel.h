// AutoDeformDebugPanel.h
//
// AutoDeform 専用のフローティング ImGui デバッグサブパネル。
//
// 目的:
//   従来 KEY (1,2,3,4,5,6,Bksp,N,-,=,7,P,0,SHIFT+1) で叩いていた
//   AutoDeform / HEMI 駆動 / Inspect / Preset の各操作を、ImGui の
//   ボタン・チェックボックスで叩けるようにする。
//
// 設計方針:
//   - DEFORM_MODE 限定。currentMainMode != DEFORM_MODE のときは何も描かない。
//   - 内部で DeformPipeline::onKey(KEY, mods) をそのまま呼ぶ。
//     ロジックを二重定義しないので、キーハンドラ側を更新すれば
//     パネル側も自動で追従する。
//   - 既存のキーボード操作は残したまま動作(共存)。動作確認後にキーは
//     消してよい。逆にパネルだけ閉じれば(右上 [X] orCollapsing) キーで
//     操作する従来運用にも戻れる。
//   - サイドパネル (RegistrationImGuiManager) の R/H/D/V/T/B/Reset 等は
//     こちらでは扱わない(別パネルとして分離)。
//
// 使い方 (main.cpp 側):
//   #include "AutoDeformDebugPanel.h"
//   ...
//   gUI.draw(gWindowWidth, gWindowHeight);
//   AutoDeformDebugPanel::draw();   // ← ここに 1 行追加
//
#pragma once

// GL ヘッダ取り込み順序衝突回避:
//   - DeformPipeline.h は内部で <GL/glew.h> を含む(mCutMesh 経由)
//   - <GLFW/glfw3.h> はデフォルトで <GL/gl.h> を引き、glew より先に gl.h
//     が定義されると glew.h 内のセーフティチェックでエラーになる
//     ("gl.h included before glew.h")。
//   - 本ヘッダでは GLFW_KEY_* 定数しか使わないので GLFW_INCLUDE_NONE で
//     GL ヘッダ取り込みを抑止する。
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include "imgui.h"

#include "DeformGlobals.h"     // gAutoDeform / gAutoCtrl / gMoveScale / etc.
#include "DeformPipeline.h"    // DeformPipeline::onKey
#include "AutoDeformOpt.h"     // Case A: autoStepActive / autoStepAll

namespace AutoDeformDebugPanel {

// パネル開閉(右上の [X] / Collapse でも閉じられる)
inline bool g_visible = true;

// ----------------------------------------------------------------------------
// 共通ヘルパ: 「KEY を 1 回タップした」のと完全に同じ挙動。
//   引数 mods は GLFW の bitmask (GLFW_MOD_SHIFT 等)。0 で素押し相当。
// ----------------------------------------------------------------------------
inline void tapKey(int key, int mods = 0) {
    if (currentMainMode != DEFORM_MODE) return;
    DeformPipeline::onKey(key, mods);
}

// ----------------------------------------------------------------------------
// 一段の押しボタン(全幅)。引数は表示ラベル + 押したときの動作。
//   有効化条件 enabled=false なら disabled スタイルで描画 + クリック無効化。
// ----------------------------------------------------------------------------
inline bool fullWidthButton(const char* label, const ImVec4& color, bool enabled = true) {
    ImVec4 c = color;
    if (!enabled) {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.6f, c.y*0.6f, c.z*0.6f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x*0.85f, c.y*0.85f, c.z*0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  c);
        ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(1, 1, 1, 1));
    }
    bool clicked = ImGui::Button(label, ImVec2(-1, 28));
    ImGui::PopStyleColor(4);
    if (!enabled) return false;
    return clicked;
}

// ============================================================================
// パネル本体
// ============================================================================
inline void draw() {
    // DEFORM_MODE 限定。Registration / その他のフェーズでは出さない。
    if (currentMainMode != DEFORM_MODE) return;

    if (!g_visible) return;

    // 初期位置(右上)。ユーザがドラッグした位置は ImGui が記憶する。
    ImGui::SetNextWindowPos(ImVec2((float)gWindowWidth - 360 - 16, 16),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.09f, 0.09f, 0.11f, 0.94f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg,         ImVec4(0.18f, 0.10f, 0.04f, 1));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive,   ImVec4(0.30f, 0.16f, 0.05f, 1));
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed,ImVec4(0.10f, 0.06f, 0.03f, 0.8f));

    bool open = true;
    ImGui::Begin("AutoDeform Debug##autodef_panel", &open,
                 ImGuiWindowFlags_NoCollapse);
    g_visible = open;   // [X] で閉じたとき次フレームから非表示

    // ------------------------------------------------------------------------
    // ステージ進行表示
    // ------------------------------------------------------------------------
    ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.20f, 1), "Stage");
    ImGui::SameLine();
    int  stage  = gAutoDeform.stage;
    bool ready5 = (stage >= 5);
    ImGui::Text("  %d / 5  %s", stage,
                gAutoDeform.fieldReady ? "(field ready)" : "");

    ImGui::Separator();

    // ------------------------------------------------------------------------
    // Stage Pipeline (旧 Key 1〜5)
    //   prerequisite: liver + target が揃っていること。
    // ------------------------------------------------------------------------
    bool prereq = (gLiverStaticMesh && gTargetMesh && multiBody);
    if (!prereq) {
        ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1),
                           "liver / target / multiBody not ready");
    }

    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.9f, 1), "STAGE PIPELINE");

    // field/handle 構築モード (debug)。classify は常に 3-way。この flag は
    // computeFieldOnVisMesh で OUTLIER 点を field に使うかどうかだけを切り替える。
    //   OFF (既定 = 3-way) : 全 ratioOK 点で field → green/red=68/73, move[0]=(-0.12108)。
    //   ON  (2-way)        : CAT_OUTLIER を field から除外 → green/red=51/87, move[0]=(0.25352)。
    //   トグル後は Stage 4 (必要なら 5) を押し直すと反映される (Stage 3 は不変)。
    ImGui::Checkbox("2-way: drop outlier from field (debug)##twoway", &gUseTwoWayClassify);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Field-build mode (Stage 4). Classification (Stage 3) is always 3-way.\n"
            "OFF (default = 3-way): field uses ALL ratioOK points (outliers included).\n"
            "    -> green/red 68/73, move[0]=(-0.12108,...). The true 3-way.\n"
            "ON (2-way): CAT_OUTLIER points are dropped from the field.\n"
            "    -> green/red 51/87, move[0]=(0.25352,...). Handle placement changes.\n"
            "Re-run Stage 4 (then 5) after toggling to apply.");
    }
    ImGui::TextDisabled("  field: %s", gUseTwoWayClassify ? "2-way (drop outlier)" : "3-way (all points)");

    if (fullWidthButton("[1] Classify Src Visibility",
                        ImVec4(0.3f, 0.55f, 0.85f, 1), prereq))
        tapKey(GLFW_KEY_1, 0);

    if (fullWidthButton("[2] Extract Correspondences",
                        ImVec4(0.3f, 0.55f, 0.85f, 1), prereq))
        tapKey(GLFW_KEY_2, 0);

    if (fullWidthButton("[3] Classify INLIER / MOVER / OUTLIER",
                        ImVec4(0.3f, 0.55f, 0.85f, 1), prereq))
        tapKey(GLFW_KEY_3, 0);

    if (fullWidthButton("[4] Compute Field on Vis Mesh",
                        ImVec4(0.3f, 0.55f, 0.85f, 1), prereq))
        tapKey(GLFW_KEY_4, 0);

    if (fullWidthButton("[5] Generate Handles + DEFORM mode",
                        ImVec4(0.95f, 0.55f, 0.20f, 1), prereq))
        tapKey(GLFW_KEY_5, 0);

    // ------------------------------------------------------------------------
    // Preset 選択 (旧 Key P)
    //   Preset 切り替え後、もし Stage 5 を一度通過していれば自動で Step 5 を
    //   再実行する。これによりスフィア(handles)数が新 preset (K_fix/K_move) に
    //   即座に反映される。Stage 5 未到達なら表示更新のみ (Step 5 押した時点で
    //   新 preset の K_fix/K_move が反映されるので問題なし)。
    // ------------------------------------------------------------------------
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.9f, 1), "PRESET");
    {
        const auto& presets = AutoDeform::getPresets();
        const auto& cur = presets[gAutoDeformPresetIdx];
        ImGui::Text("Current: %s", cur.name);
        ImGui::Text("  K_fix=%d  K_move=%d  rFix=%.2f  rMove=%.2f",
                    cur.K_fix, cur.K_move, cur.rFixScale, cur.rMoveScale);

        if (fullWidthButton("Next Preset (cycle P0..P4)",
                            ImVec4(0.45f, 0.5f, 0.75f, 1), true)) {
            tapKey(GLFW_KEY_P, 0);
            // Stage 5 を通過済みなら自動で再生成 → スフィア数が即時更新
            if (gAutoDeform.stage >= 5) {
                tapKey(GLFW_KEY_5, 0);
            }
        }

        if (gAutoDeform.stage >= 5) {
            ImGui::TextDisabled("  (handles auto-regenerated on preset change)");
        } else {
            ImGui::TextDisabled("  (will apply when you press Stage 5)");
        }
    }

    // ------------------------------------------------------------------------
    // HEMI Drive (旧 Key 6 / Bksp / N / - / =)
    //   prerequisite: ハンドル生成済み (stage >= 5)
    // ------------------------------------------------------------------------
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.9f, 1), "HEMI DRIVE");

    bool hasHandles = ready5 && gAutoCtrl.numMove() > 0;
    int activeIdx   = hasHandles ? gAutoCtrl.activeMoveIdx() : -1;

    if (hasHandles) {
        ImGui::Text("Active move: %d / %d", activeIdx, gAutoCtrl.numMove());
        if (activeIdx >= 0) {
            float p = gAutoCtrl.moveHandle(activeIdx).progress;
            ImGui::ProgressBar(p, ImVec2(-1, 0), nullptr);
        }
    } else {
        ImGui::TextDisabled("Run [5] first to generate handles");
    }

    // Step+ / Step-  (2 列)
    {
        float bw = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        // ★ fullWidthButton は -1 幅なので使えず、ローカルで色付け
        auto halfBtn = [&](const char* label, ImVec4 c, bool enabled) -> bool {
            if (!enabled) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.6f, c.y*0.6f, c.z*0.6f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x*0.85f, c.y*0.85f, c.z*0.85f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  c);
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(1, 1, 1, 1));
            }
            bool cl = ImGui::Button(label, ImVec2(bw, 28));
            ImGui::PopStyleColor(4);
            return enabled && cl;
        };
        if (halfBtn("Step + (6)",  ImVec4(0.35f, 0.75f, 0.40f, 1), hasHandles && activeIdx >= 0))
            tapKey(GLFW_KEY_6, 0);
        ImGui::SameLine();
        if (halfBtn("Step - (Bksp)", ImVec4(0.85f, 0.55f, 0.25f, 1), hasHandles && activeIdx >= 0))
            tapKey(GLFW_KEY_BACKSPACE, 0);
    }

    if (fullWidthButton("Next Move Handle (N)",
                        ImVec4(0.45f, 0.55f, 0.75f, 1), hasHandles))
        tapKey(GLFW_KEY_N, 0);

    // MoveScale (旧 - / =)
    ImGui::Text("Move Scale: %.2f", gMoveScale);
    {
        float bw = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        if (ImGui::Button("-0.1", ImVec2(bw, 22))) tapKey(GLFW_KEY_MINUS, 0);
        ImGui::SameLine();
        if (ImGui::Button("+0.1", ImVec2(bw, 22))) tapKey(GLFW_KEY_EQUAL, 0);
    }
    ImGui::TextDisabled("  (range 0.1..2.0)");

    // ------------------------------------------------------------------------
    // AUTO OPTIMIZE (Case A: stateful patience-based step search)
    //   - "Auto OPTIMIZE this handle" : 現アクティブ handle を最適化
    //   - "Auto OPTIMIZE all handles" : 全 move handles を round-robin
    //   - "Reset progress -> 0"       : 全 handle progress を 0 まで巻き戻し
    //   - Two-stage checkbox          : Coarse → Fine 2 段階探索 ON/OFF
    //   Settings expander で patience / eps / boost iter / stage scales を露出。
    // ------------------------------------------------------------------------
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.20f, 1), "AUTO OPTIMIZE  (Case A)");

    {
        float bw = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        auto optBtn = [&](const char* lbl, ImVec4 c, bool en, float w) -> bool {
            if (!en) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(c.x*0.25f, c.y*0.25f, c.z*0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.6f, c.y*0.6f, c.z*0.6f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x*0.85f, c.y*0.85f, c.z*0.85f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  c);
                ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(1, 1, 1, 1));
            }
            bool cl = ImGui::Button(lbl, ImVec2(w, 32));
            ImGui::PopStyleColor(4);
            return en && cl;
        };
        // Row 1: Optimize this / Optimize all
        if (optBtn("Auto OPTIMIZE this handle",
                   ImVec4(0.20f, 0.75f, 0.40f, 1),
                   hasHandles && activeIdx >= 0, bw)) {
            AutoDeformOpt::autoStepActive();
        }
        ImGui::SameLine();
        if (optBtn("Auto OPTIMIZE all handles",
                   ImVec4(0.15f, 0.55f, 0.85f, 1), hasHandles, bw)) {
            AutoDeformOpt::autoStepAll();
        }

        // Row 2: Reset all progress (full width)
        float fullW = ImGui::GetContentRegionAvail().x;
        if (optBtn("Reset all progress -> 0",
                   ImVec4(0.75f, 0.45f, 0.25f, 1), hasHandles, fullW)) {
            AutoDeformOpt::resetAllProgress();
        }
    }

    // Two-stage toggle (debug 用): ON で Coarse(0.25) -> Fine(0.05) 探索になる
    ImGui::Checkbox("Two-stage (coarse -> fine)##2stage",
                    &AutoDeformOpt::gUseTwoStage);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "OFF: 1-stage search using current gMoveScale (default 0.1).\n"
            "ON : Coarse pass (scale=%.2f) -> rewind 1 coarse step ->\n"
            "     Fine pass (scale=%.2f).  Slower but tends to find\n"
            "     lower RMSE because the coarse pass quickly locates\n"
            "     the basin and the fine pass refines inside it.\n"
            "Tunable in Settings below.",
            AutoDeformOpt::gCoarseScale, AutoDeformOpt::gFineScale);
    }

    // 最新結果ステータス表示
    if (AutoDeformOpt::gLastSingle.moveIdx >= 0) {
        const auto& s = AutoDeformOpt::gLastSingle;
        ImGui::TextDisabled("Last single: m=%d steps=%d(-%d) %.5f -> %.5f%s (%s)",
                            s.moveIdx, s.stepsTaken, s.stepsUnwound,
                            s.startRMSE, s.bestRMSE,
                            s.twoStage ? " [2-stage]" : "",
                            AutoDeformOpt::toStr(s.stopReason));
    }
    if (AutoDeformOpt::gLastAll.passes > 0) {
        const auto& a = AutoDeformOpt::gLastAll;
        float imp = a.startRMSE - a.bestRMSE;
        ImGui::TextDisabled("Last all: %d pass, %d steps  %.5f -> %.5f  (%+.5f)%s  %s",
                            a.passes, a.totalSteps,
                            a.startRMSE, a.bestRMSE, -imp,
                            a.twoStage ? " [2-stage]" : "",
                            AutoDeformOpt::toStr(a.stopReason));
    }

    // Settings expander (default 折り畳み)
    if (ImGui::TreeNode("Settings##autoopt")) {
        ImGui::TextDisabled("Patience model:");
        ImGui::SliderInt  ("Max steps / handle",       &AutoDeformOpt::gMaxSteps,    1,   50);
        ImGui::SliderInt  ("Patience (no-improve N)",  &AutoDeformOpt::gPatience,    1,   5);
        ImGui::SliderFloat("Eps ratio (rel. improve)", &AutoDeformOpt::gEpsRatio,
                           0.0001f, 0.01f, "%.4f");
        ImGui::SliderInt  ("Max passes (Auto ALL)",    &AutoDeformOpt::gMaxPasses,   1,   10);

        ImGui::Separator();
        ImGui::TextDisabled("Quality:");
        ImGui::SliderInt  ("Boost iter during auto",   &AutoDeformOpt::gAutoBoostIter,
                           20, 200);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Number of solver iterations per Key 6 / Bksp tap, while\n"
                "auto is running. Higher = immediate RMSE closer to settled\n"
                "RMSE = more accurate patience decisions. Manual Key 6 uses\n"
                "gAutoDeformBoostIter (default 30).");
        }

        ImGui::Separator();
        ImGui::TextDisabled("Two-stage (coarse -> fine):");
        ImGui::SliderFloat("Coarse scale", &AutoDeformOpt::gCoarseScale, 0.10f, 0.50f, "%.2f");
        ImGui::SliderFloat("Fine scale",   &AutoDeformOpt::gFineScale,   0.01f, 0.20f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fine scale should be < Coarse scale.\n"
                              "Typical: coarse=0.25, fine=0.05.");
        }
        ImGui::TreePop();
    }

    // ------------------------------------------------------------------------
    // FINE-TUNE (AUTO 後の手動微調整)
    //   ON にすると DEFORM_MODE の通常メッシュグラブを止め、AUTO ハンドル球
    //   (fix + move 両方) を直接クリックして掴み、カメラ平面内 (奥行き固定) で
    //   ドラッグできる。fix/move は区別しない。各 handle の manualOffset を更新し、
    //   リアルタイムで物理が追従する。AUTO の progress は保持されるので
    //   「AUTO 結果の上に手で補正を足す」運用。
    // ------------------------------------------------------------------------
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.75f, 1), "FINE-TUNE  (manual nudge)");
    {
        bool ftReady = (multiBody && gAutoCtrl.numHandles() > 0);

        if (!ftReady) {
            // ハンドル未生成なら使えない。フラグも倒しておく。
            gFineTuneMode = false;
            ImGui::TextDisabled("Run Stage 5 (generate handles) first.");
        } else {
            ImGui::Checkbox("Fine-tune mode (grab any sphere)", &gFineTuneMode);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON: left-click any AUTO handle sphere (fix or move) and drag\n"
                    "    to nudge it. fix and move are treated the same.\n"
                    "    Movement stays in the current camera plane (depth fixed).\n"
                    "    Rotate the camera to nudge from another angle; in AR mode\n"
                    "    (Key A) the drag plane follows the AR view.\n"
                    "    The mesh follows in real time. AUTO progress is preserved.\n"
                    "OFF: normal DEFORM mesh grab is active.");
            }

            if (gFineTuneMode) {
                ImGui::TextColored(ImVec4(0.95f, 0.7f, 0.4f, 1),
                                   "  ACTIVE: drag any sphere (fix/move) to nudge");
            }

            // Clear manual offsets: 手動補正だけ 0 に戻す (AUTO progress は残る)。
            bool hasOffsets = gAutoCtrl.hasAnyManualOffset();
            if (fullWidthButton("Clear manual offsets",
                                ImVec4(0.75f, 0.45f, 0.55f, 1), hasOffsets)) {
                gAutoCtrl.clearManualOffsets(multiBody);
                // 物理を馴染ませて offset 除去を反映。
                gAutoCtrl.runBoost(multiBody, gAutoDeformBoostIter, gAutoDeformBoostDamping);
            }
            if (!hasOffsets) {
                ImGui::TextDisabled("  (no manual offsets to clear)");
            }
        }
    }
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.9f, 1), "INSPECT / DEBUG");

    // Key 7: BEFORE/AFTER snapshot toggle
    {
        const char* lbl = "Toggle BEFORE/AFTER snapshot (7)";
        if (gShowingAfter) lbl = "Currently: AFTER   [press to swap]";
        else if (gSnapBeforeValid) lbl = "Currently: BEFORE  [press to swap]";

        bool canSnap = (multiBody && gTargetMesh);
        if (fullWidthButton(lbl,
                            ImVec4(0.85f, 0.40f, 0.85f, 1),
                            canSnap))
            tapKey(GLFW_KEY_7, 0);

        // 状態の補足表示
        ImGui::TextDisabled("  before=%s  after=%s  inspect=%s",
                            gSnapBeforeValid ? "yes" : "no",
                            gSnapAfterValid  ? "yes" : "no",
                            gInspectMode     ? "ON"  : "off");
    }

    // Key 0: TetMesh wireframe toggle
    {
        bool wf = multiBody && multiBody->isTetMeshVisible();
        if (ImGui::Checkbox("TetMesh wireframe (0)", &wf)) {
            tapKey(GLFW_KEY_0, 0);
        }
    }

    // SHIFT+Key1: single sphere debug toggle
    {
        bool single = (gAutoDeform.debugMaxPoints != 0);
        if (ImGui::Checkbox("Single sphere debug (SHIFT+1/2/3)", &single)) {
            // SHIFT+1 は debugMaxPoints をトグルしてパイプライン再実行なし
            tapKey(GLFW_KEY_1, GLFW_MOD_SHIFT);
        }
    }

    // ------------------------------------------------------------------------
    // Status (RMSE / shape stats)
    // ------------------------------------------------------------------------
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.9f, 1), "STATUS");
    {
        // 手動 Calculate: 今の変形状態 (手で動かした handle / auto 結果問わず) で
        // VisMesh 0 の RMSE を即計測する。auto 駆動とは独立して「今の見た目の
        // RMSE が知りたい」ときに使う。結果は static に保持してログにも詳細を出す。
        static float s_calcRMSE   = -1.0f;
        static bool  s_calcValid  = false;

        bool canCalc = (multiBody && gTargetMesh);
        if (fullWidthButton("Calculate RMSE now",
                            ImVec4(0.30f, 0.70f, 0.55f, 1), canCalc)) {
            const auto& visVerts = multiBody->getVisPositions(0);
            s_calcRMSE = DeformPipeline::autoMeasureRMSE(
                visVerts, gTargetMesh,
                /*storeAsBefore=*/false,
                DeformPipeline::scaledMaxDist(),
                /*verbose=*/true);   // ログに target/matched/Avg/Max 詳細を出す
            s_calcValid = (s_calcRMSE >= 0.0f);
        }
        if (s_calcValid) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.6f, 1),
                               "  -> RMSE = %.5f", s_calcRMSE);
        }

        ImGui::Spacing();
        ImGui::Text("RMSE before: %s",
                    gAutoDeform.rmseBeforeDeform >= 0.0f
                        ? std::to_string(gAutoDeform.rmseBeforeDeform).c_str()
                        : "-");
        ImGui::Text("RMSE last  : %s",
                    gAutoDeform.rmseLastMeasured >= 0.0f
                        ? std::to_string(gAutoDeform.rmseLastMeasured).c_str()
                        : "-");
        ImGui::Text("Correspondences: %zu",
                    gAutoDeform.correspondences.size());
        ImGui::Text("fix handles: %zu  move handles: %zu",
                    gAutoDeform.fixHandles.size(),
                    gAutoDeform.moveHandles.size());
    }

    ImGui::End();
    ImGui::PopStyleColor(4);
}

}  // namespace AutoDeformDebugPanel
