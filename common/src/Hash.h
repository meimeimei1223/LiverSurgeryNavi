#ifndef HASH_H
#define HASH_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

class Hash {
public:
    float spacing;
    int tableSize;
    std::vector<int> cellStart;
    std::vector<int> cellEntries;
    std::vector<int> queryIds;
    int querySize;
    int numPoints;  // 追加：点の総数を保持
    /*
    Hash(float spacing, int maxNumObjects)
        : spacing(spacing), tableSize(2 * maxNumObjects), querySize(0) {
        cellStart.resize(tableSize + 1, 0);
        cellEntries.resize(maxNumObjects, 0);
        queryIds.resize(maxNumObjects, 0);
    }
    */
    Hash(float spacing, int maxNumObjects)
        : spacing(spacing)
        , tableSize(2 * maxNumObjects)
        , querySize(0)
        , numPoints(maxNumObjects) {  // numPointsの初期化を追加
        cellStart.resize(tableSize + 1, 0);
        cellEntries.resize(maxNumObjects, 0);
        queryIds.resize(maxNumObjects, 0);
    }

    // ハッシュ関数
    int hashCoords(int xi, int yi, int zi) {
        int h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481); // ファンタジー関数
        return std::abs(h) % tableSize;
    }

    // 座標を整数セルに変換
    int intCoord(float coord) {
        return static_cast<int>(std::floor(coord / spacing));
    }

    // 位置情報をハッシュ値に変換
    int hashPos(const std::vector<float>& pos, int nr) {
        return hashCoords(
            intCoord(pos[3 * nr]),
            intCoord(pos[3 * nr + 1]),
            intCoord(pos[3 * nr + 2])
            );
    }

    // ハッシュテーブルの作成
    void create(const std::vector<float>& pos) {
        int numObjects = std::min(static_cast<int>(pos.size() / 3), static_cast<int>(cellEntries.size()));

        std::fill(cellStart.begin(), cellStart.end(), 0);
        std::fill(cellEntries.begin(), cellEntries.end(), 0);

        for (int i = 0; i < numObjects; i++) {
            int h = hashPos(pos, i);
            cellStart[h]++;
        }

        // セルの開始位置を決定
        int start = 0;
        for (int i = 0; i < tableSize; i++) {
            start += cellStart[i];
            cellStart[i] = start;
        }
        cellStart[tableSize] = start; // ガード

        // オブジェクトIDを格納
        for (int i = 0; i < numObjects; i++) {
            int h = hashPos(pos, i);
            cellStart[h]--;
            cellEntries[cellStart[h]] = i;
        }
    }
    void query(const std::vector<float>& pos, int nr, float maxDist) {
        // [scale-fix] 検索距離の上限は「セル数」で規定する(=単位非依存)。
        //   旧コードは absolute 5.0 にクランプしていた。これは registration の
        //   メートル系フレーム(メッシュ径 ~0.7、tet 外接半径 < 5)では無害だが、
        //   CT-handoff(reg_transform.txt)導入後、DEFORM は臓器を CT-mm 系
        //   (肝臓径 ~305mm、tet 外接半径=数十mm)で扱うため maxDist > 5 が常時成立
        //   し、スキニングの全 tet (数千個)で "Search distance limited" を std::endl
        //   連発 → stdout フラッシュでスキニングが極端に遅く(実質ハング)なっていた。
        //   spacing(=メッシュ径基準)に紐付ければ m / mm どちらのスケールでも破綻
        //   せず、メートル系での挙動(警告なし)も従来どおり維持される。
        const int   MAX_CELL_RANGE = 20;                       // 反復セル数の安全上限
        const float maxAllowed     = MAX_CELL_RANGE * spacing * 0.5f; // ±maxDist で 20 セル以内
        if (spacing > 0.0f && maxDist > maxAllowed) {
            static bool s_warnedClamp = false;
            if (!s_warnedClamp) {
                std::cout << "[Hash] note: search distance clamped to " << maxAllowed
                          << " (spacing=" << spacing
                          << "); further clamp messages suppressed." << std::endl;
                s_warnedClamp = true;
            }
            maxDist = maxAllowed;
        }

        // セルの範囲を計算
        int x0 = intCoord(pos[3 * nr] - maxDist);
        int y0 = intCoord(pos[3 * nr + 1] - maxDist);
        int z0 = intCoord(pos[3 * nr + 2] - maxDist);
        int x1 = intCoord(pos[3 * nr] + maxDist);
        int y1 = intCoord(pos[3 * nr + 1] + maxDist);
        int z1 = intCoord(pos[3 * nr + 2] + maxDist);

        // セル範囲の最終ガード(上の clamp により通常は発火しない)。
        if (x1 - x0 > MAX_CELL_RANGE || y1 - y0 > MAX_CELL_RANGE || z1 - z0 > MAX_CELL_RANGE) {
            static bool s_warnedRange = false;
            if (!s_warnedRange) {
                std::cout << "[Hash] note: cell range too large, search limited; "
                             "further messages suppressed." << std::endl;
                s_warnedRange = true;
            }
            querySize = 0;
            return;
        }


        querySize = 0;

        for (int xi = x0; xi <= x1; xi++) {
            for (int yi = y0; yi <= y1; yi++) {
                for (int zi = z0; zi <= z1; zi++) {
                    int h = hashCoords(xi, yi, zi);
                    if (h < 0 || h >= tableSize) continue;

                    int start = cellStart[h];
                    int end = cellStart[h + 1];

                    for (int i = start; i < end && querySize < queryIds.size(); i++) {
                        int id = cellEntries[i];
                        if (id >= 0 && id < numPoints) {  // 範囲チェックを追加
                            queryIds[querySize++] = id;
                        }
                    }
                }
            }
        }
    }



    /*
    // 指定範囲内のオブジェクトをクエリ
    void query(const std::vector<float>& pos, int nr, float maxDist) {
        int x0 = intCoord(pos[3 * nr] - maxDist);
        int y0 = intCoord(pos[3 * nr + 1] - maxDist);
        int z0 = intCoord(pos[3 * nr + 2] - maxDist);

        int x1 = intCoord(pos[3 * nr] + maxDist);
        int y1 = intCoord(pos[3 * nr + 1] + maxDist);
        int z1 = intCoord(pos[3 * nr + 2] + maxDist);

        querySize = 0;

        for (int xi = x0; xi <= x1; xi++) {
            for (int yi = y0; yi <= y1; yi++) {
                for (int zi = z0; zi <= z1; zi++) {
                    int h = hashCoords(xi, yi, zi);
                    int start = cellStart[h];
                    int end = cellStart[h + 1];

                    for (int i = start; i < end; i++) {
                        queryIds[querySize++] = cellEntries[i];
                    }
                }
            }
        }
    }
    */
};

#endif // HASH_H
