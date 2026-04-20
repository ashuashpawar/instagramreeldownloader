Version: v1.0 — Face Swap Working
Saved: 2026-04-18

What works in this version:
- Face swap (InsightFace + inswapper_128)
- Hero face selection via embedding similarity (multi-face videos)
- EMA bounding box smoothing (eliminates flicker)
- Reinhard colour match (skin-tone alignment)
- Feathered alpha blend (zero-seam edges)
- Multi-pass sharpening (counters 128px GAN softness)
- GFPGAN face enhancement pass (post-swap sharpening)
- CodeFormer ONNX support with safe load + auto-delete on corrupt file
- BGR→RGB fix for GFPGAN (prevents blue face tint)
- Native macOS file/folder pickers everywhere
- Auto-increment output naming (never overwrites)
- Workflow recording & replay (agent learns sessions)
- All Replicate cloud options removed and menu renumbered

To restore this version:
  cp versions/v1.0_face_swap_working/video_editor.py .
  cp versions/v1.0_face_swap_working/agent.py .
