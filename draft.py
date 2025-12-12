def _analyze_board_best(img_pil: "Image.Image", debug_dir: Optional[str] = None) -> Tuple[List[str], float]:
    """
    先用掩码定位“黄色最多的格”（最多取前3个），再对这些格的原图做 OCR。
    若 OCR 全部失败，则以“面积最大”的格为 best_move，net_win=0.0（fallback）。
    """
    import numpy as np
    import cv2
    from PIL import Image

    # 1) 定位：按每格黄像素面积排序
    candidates, mask = _detect_best_cells_by_mask(img_pil, top_k=None, debug_dir=debug_dir)
    if not candidates:
        raise RuntimeError("no yellow found on board (mask empty)")

    # 2) 依次对前 K 个格做 OCR，选读到的最大值；并列≤0.1按字典序
    results = []  # [(coord, val, rect)]
    for idx, (coord, area, rect) in enumerate(candidates):
        subdir = os.path.join(debug_dir, f"roi_{idx:02d}") if debug_dir else None
        val, raw = _ocr_cell_value(img_pil, rect, debug_dir=subdir, tag=f"{coord}")
        if val is not None:
            results.append((coord, float(val), rect))

    if results:
        # 合并同格（理论上 candidates 已唯一）
        results.sort(key=lambda t: t[1], reverse=True)
        best_val = results[0][1]
        tied = [r for r in results if abs(r[1] - best_val) <= 0.1 + 1e-9]
        tied.sort(key=lambda t: t[0])  # 字典序
        best_move = tied[0][0]
        net_win = float(best_val)
    else:
        # 3) 兜底：OCR 全失败 → 选择黄色面积最大的格，net_win=0.0
        best_move = candidates[0][0]
        net_win = 0.0
        if debug_dir:
            with open(os.path.join(debug_dir, "fallback.txt"), "w", encoding="utf-8") as f:
                f.write("fallback:no-ocr; choose max-area cell\n")

    # 4) 调试可视化
    if debug_dir:
        import numpy as np
        img_rgb = np.array(img_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        base = os.path.join(debug_dir, "sensei")
        cv2.imwrite(base + "_mask.png", mask)
        vis = img_bgr.copy()
        for coord, area, (x0, y0, x1, y1) in candidates:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
            cv2.putText(vis, f"{coord}:{area}", (x0, max(10, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(base + "_vis.png", vis)

    return [best_move], net_win