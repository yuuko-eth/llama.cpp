#pragma once

// Parallel Box Decoding (PBD) support for NVIDIA LocateAnything-style grounders.
//
// LocateAnything emits bounding boxes as fixed-shape token blocks
// (<box><x1><y1><x2><y2></box>). Instead of decoding those coordinate tokens one
// at a time, the model is trained to fill a block of <text_mask> placeholders in a
// single forward pass (Multi-Token Prediction). This header exposes the pure
// decode-side helpers; the actual decode loop lives in mtmd-cli.cpp because it
// needs the llama_context / KV-cache plumbing.
//
// Reference (NVIDIA): modeling_locateanything.py (generate loop) and
// generate_utils.py (sample_tokens / decode_bbox_avg / handle_pattern). This is a
// from-scratch C++ port for the greedy (temperature 0) path that grounding uses.

#include "llama.h"

#include <vector>

enum class grounding_mode {
    SLOW,   // pure autoregressive — the existing token-by-token path
    FAST,   // MTP only, never falls back to AR
    HYBRID, // MTP first, fall back to AR on a malformed box, switch back on </box>
};

// Read MTMD_GROUNDING_MODE ("slow" | "fast" | "hybrid"). Defaults to SLOW so the
// CLI behaves exactly as before unless grounding is explicitly requested.
grounding_mode grounding_mode_from_env();
const char *   grounding_mode_name(grounding_mode m);

// The control token IDs that drive PBD. Resolved from the model vocab by string at
// startup (they ship as added tokens in the LocateAnything GGUF). A couple of plain
// vocab tokens that have no stable surface form fall back to documented constants.
struct grounding_tokens {
    llama_token mask        = -1; // <text_mask>
    llama_token null_tok    = -1; // <null>
    llama_token none_tok    = -1; // "none" (empty-box content token)
    llama_token box_start   = -1; // <box>
    llama_token box_end     = -1; // </box>
    llama_token coord_start = -1; // <0>     — lowest coordinate token
    llama_token coord_end   = -1; // <1000>  — highest coordinate token
    llama_token ref_start   = -1; // <ref>
    llama_token ref_end     = -1; // </ref>
    llama_token im_end      = -1; // <|im_end|>

    int n_future = 6; // block size: one box (<box> + 4 coords + </box>)

    // Resolve all IDs. Returns false (and logs) if a required token is missing,
    // which means the loaded model is not a LocateAnything-style grounder.
    bool resolve(const llama_vocab * vocab);

    bool is_coord(llama_token t) const { return t >= coord_start && t <= coord_end; }
};

// Classification of one MTP block, mirroring the reference handle_pattern().
enum class mtp_pattern {
    IM_END,     // terminate generation
    EMPTY_BOX,  // <box>none</box>
    COORD_BOX,  // <box><x1><y1><x2><y2></box>
    POINT_BOX,  // <box><x><y></box>
    ERROR_BOX,  // malformed box — HYBRID switches to AR, FAST treats as coord box
    REF_OBJECT, // <ref> ... label text ...
};

struct mtp_step_result {
    mtp_pattern              pattern;
    std::vector<llama_token> tokens;            // tokens to actually emit this step
    bool                     need_switch_to_ar = false;
    bool                     is_terminal       = false;
};

// Greedy MTP step: given the n_future logit rows produced by one non-causal window
// decode (each row is a [n_vocab] pointer from llama_get_logits_ith), reproduce
// sample_tokens() + decode_bbox_avg()/decode_ref() + handle_pattern() and return the
// tokens to emit. `mode` selects the FAST vs HYBRID behaviour on malformed boxes.
mtp_step_result mtp_decode_step(const std::vector<const float *> & logit_rows,
                                int                                n_vocab,
                                const grounding_tokens &           tok,
                                grounding_mode                     mode);
