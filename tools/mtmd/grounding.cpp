#include "grounding.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

grounding_mode grounding_mode_from_env() {
    const char * e = std::getenv("MTMD_GROUNDING_MODE");
    if (!e) {
        return grounding_mode::SLOW;
    }
    if (std::strcmp(e, "fast")   == 0) return grounding_mode::FAST;
    if (std::strcmp(e, "hybrid") == 0) return grounding_mode::HYBRID;
    return grounding_mode::SLOW;
}

const char * grounding_mode_name(grounding_mode m) {
    switch (m) {
        case grounding_mode::FAST:   return "fast";
        case grounding_mode::HYBRID: return "hybrid";
        default:                     return "slow";
    }
}

static llama_token resolve_special(const llama_vocab * vocab, const char * text) {
    // A registered special/added token tokenizes to exactly one id.
    std::vector<llama_token> toks = common_tokenize(vocab, text, false, true);
    return toks.size() == 1 ? toks[0] : -1;
}

bool grounding_tokens::resolve(const llama_vocab * vocab) {
    mask        = resolve_special(vocab, "<text_mask>");
    null_tok    = resolve_special(vocab, "<null>");
    box_start   = resolve_special(vocab, "<box>");
    box_end     = resolve_special(vocab, "</box>");
    coord_start = resolve_special(vocab, "<0>");
    coord_end   = resolve_special(vocab, "<1000>");
    ref_start   = resolve_special(vocab, "<ref>");
    ref_end     = resolve_special(vocab, "</ref>");

    im_end = resolve_special(vocab, "<|im_end|>");
    if (im_end < 0) {
        im_end = llama_vocab_eos(vocab);
    }

    // "none" is a plain word token (the empty-box content token); it has no stable
    // angle-bracket form, so we use the LocateAnything vocab constant. Only affects
    // <box>none</box> detection if it ever drifts.
    none_tok = 4064;

    const bool ok = mask >= 0 && null_tok >= 0 && box_start >= 0 && box_end >= 0 &&
                    coord_start >= 0 && coord_end >= 0 && ref_start >= 0 && ref_end >= 0;
    if (!ok) {
        LOG_ERR("%s: model is missing LocateAnything grounding tokens "
                "(mask=%d box=[%d,%d] coord=[%d,%d] ref=[%d,%d] null=%d) — "
                "grounding modes require a LocateAnything-style model\n",
                __func__, mask, box_start, box_end, coord_start, coord_end,
                ref_start, ref_end, null_tok);
    }
    return ok;
}

namespace {

// A softmax view over one logit row, with cheap top-k and per-token probability.
struct row_dist {
    const float * logits = nullptr;
    int   n      = 0;
    float maxl   = 0.0f;
    float sumexp = 0.0f;

    void init(const float * l, int nv) {
        logits = l;
        n      = nv;
        maxl   = l[0];
        for (int i = 1; i < nv; ++i) {
            if (l[i] > maxl) maxl = l[i];
        }
        double s = 0.0;
        for (int i = 0; i < nv; ++i) {
            s += std::exp((double) (l[i] - maxl));
        }
        sumexp = (float) s;
    }

    float prob(llama_token t) const {
        return std::exp(logits[t] - maxl) / sumexp;
    }

    llama_token argmax() const {
        llama_token best = 0;
        float       bv   = logits[0];
        for (int i = 1; i < n; ++i) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        return best;
    }

    // top-k token ids by logit (descending), paired with their probabilities
    std::vector<std::pair<llama_token, float>> topk(int k) const {
        k = std::min(k, n);
        std::vector<llama_token> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](llama_token a, llama_token b) { return logits[a] > logits[b]; });
        std::vector<std::pair<llama_token, float>> out;
        out.reserve(k);
        for (int i = 0; i < k; ++i) {
            out.emplace_back(idx[i], prob(idx[i]));
        }
        return out;
    }
};

// Port of is_valid_box_frame + decode_bbox_avg. Returns the assembled 6-token box
// ([<box>, c1, c2, c3, c4, </box>]) or an empty vector if these rows are not a box.
std::vector<llama_token> decode_bbox_avg(const std::vector<row_dist> & rows,
                                         const grounding_tokens &      tok,
                                         grounding_mode                mode) {
    const float start_thresh = 0.6f;
    const float end_thresh   = 0.2f;
    const int   keep_k       = 5;

    // empty box: <box> none </box> <null> <null>
    if (rows[0].prob(tok.box_start) >= start_thresh &&
        rows[1].prob(tok.none_tok)  > 0.2f &&
        rows[2].prob(tok.box_end)   > 0.2f &&
        rows[3].prob(tok.null_tok)  > 0.1f &&
        rows[4].prob(tok.null_tok)  > 0.1f) {
        return { tok.box_start, tok.none_tok, tok.box_end, tok.null_tok, tok.null_tok, tok.null_tok };
    }

    // the block must plausibly end (</box> / <null> / <|im_end|>) at the last slot
    const float end_score = rows[5].prob(tok.box_end) +
                            rows[5].prob(tok.null_tok) +
                            rows[5].prob(tok.im_end);
    if (end_score < end_thresh) {
        return {}; // illegal box
    }

    llama_token coords[4];
    for (int p = 0; p < 4; ++p) {
        const auto tk = rows[1 + p].topk(keep_k);

        std::vector<std::pair<llama_token, float>> valid;
        for (const auto & e : tk) {
            if (tok.is_coord(e.first)) valid.push_back(e);
        }
        if (valid.empty()) {
            return {}; // a coordinate slot has no coordinate token in its top-k
        }

        const llama_token first_valid_id   = valid[0].first;
        const float       first_valid_prob = valid[0].second;

        if (mode == grounding_mode::HYBRID) {
            // Low-confidence + high-disagreement coordinate → mark uncertain (0),
            // which forces an error_box downstream and an AR re-decode of this box.
            llama_token vmax = valid[0].first, vmin = valid[0].first;
            for (const auto & e : valid) {
                vmax = std::max(vmax, e.first);
                vmin = std::min(vmin, e.first);
            }
            const bool is_abnormal = (first_valid_prob < 0.9f) &&
                                     ((int) valid.size() > 1) &&
                                     ((vmax - vmin) > 60);
            coords[p] = is_abnormal ? 0 : first_valid_id;
        } else {
            coords[p] = first_valid_id;
        }
    }

    return { tok.box_start, coords[0], coords[1], coords[2], coords[3], tok.box_end };
}

// Port of decode_ref: <ref> + the highest-probability non-coordinate token at each
// subsequent slot. Returns empty if row 0 is not confidently <ref>.
std::vector<llama_token> decode_ref(const std::vector<row_dist> & rows,
                                    const grounding_tokens &      tok) {
    const float start_thresh = 0.6f;
    const int   keep_k       = 5;

    if (rows[0].prob(tok.ref_start) < start_thresh) {
        return {};
    }

    std::vector<llama_token> out = { tok.ref_start };
    for (size_t p = 1; p < rows.size(); ++p) {
        const auto  tk     = rows[p].topk(keep_k);
        llama_token chosen = -1;
        for (const auto & e : tk) {
            if (!tok.is_coord(e.first)) { chosen = e.first; break; }
        }
        if (chosen < 0) {
            return {}; // a slot had only coordinate tokens in its top-k
        }
        out.push_back(chosen);
    }
    return out;
}

// Port of handle_pattern: classify the chosen block and produce the emit list.
mtp_step_result handle_pattern(const std::vector<llama_token> & x0,
                               const grounding_tokens &         tok,
                               grounding_mode                   mode) {
    mtp_step_result r;
    auto X = [&](size_t i) -> llama_token { return i < x0.size() ? x0[i] : tok.null_tok; };

    if (X(0) == tok.null_tok || X(0) == tok.im_end) {
        r.pattern     = mtp_pattern::IM_END;
        r.tokens      = { tok.im_end };
        r.is_terminal = true;
        return r;
    }

    if (X(0) == tok.box_start && X(1) == tok.none_tok) {
        r.pattern = mtp_pattern::EMPTY_BOX;
        r.tokens  = { tok.box_start, tok.none_tok, tok.box_end };
        return r;
    }

    if (X(0) == tok.box_start) {
        int coord_ix = 1;
        for (int i = 1; i <= 4; ++i) {
            if (tok.is_coord(X(i))) coord_ix++;
            else break;
        }

        if (coord_ix == 5 && X(5) == tok.box_end) {
            r.pattern = mtp_pattern::COORD_BOX;
            r.tokens  = { X(0), X(1), X(2), X(3), X(4), X(5) };
            return r;
        }
        if (coord_ix == 3 && X(3) == tok.box_end) {
            r.pattern = mtp_pattern::POINT_BOX;
            r.tokens  = { X(0), X(1), X(2), X(3) };
            return r;
        }
        if (mode == grounding_mode::FAST) {
            r.pattern = mtp_pattern::COORD_BOX;
            r.tokens  = { X(0), X(1), X(2), X(3), X(4), X(5) };
            return r;
        }
        // hybrid: malformed box → emit the valid coordinate prefix and switch to AR
        r.pattern           = mtp_pattern::ERROR_BOX;
        r.tokens.assign(x0.begin(), x0.begin() + std::min((size_t) coord_ix, x0.size()));
        r.need_switch_to_ar = true;
        return r;
    }

    // <ref> ... label text ...: keep through the first </ref>. The reference relies
    // on <null> / doubled </ref> padding to bound the span, but this model continues
    // with plain text after </ref>, so we stop at the closing tag explicitly.
    std::vector<llama_token> t;
    for (llama_token v : x0) {
        if (v == tok.null_tok) break;
        t.push_back(v);
        if (v == tok.ref_end) break;
    }
    r.pattern = mtp_pattern::REF_OBJECT;
    r.tokens  = t;
    return r;
}

} // namespace

mtp_step_result mtp_decode_step(const std::vector<const float *> & logit_rows,
                                int                                n_vocab,
                                const grounding_tokens &           tok,
                                grounding_mode                     mode) {
    const int N = (int) logit_rows.size();

    std::vector<row_dist> rows(N);
    for (int i = 0; i < N; ++i) {
        rows[i].init(logit_rows[i], n_vocab);
    }

    // greedy per-slot tokens (temperature 0 — the path grounding uses)
    std::vector<llama_token> x0(N);
    for (int i = 0; i < N; ++i) {
        x0[i] = rows[i].argmax();
    }

    // If the model's greedy first slot is a stop token, honour it. The box-averaging
    // step below is only meant to *refine* a box the model is actually emitting; when
    // the model wants to stop it pads the coordinate slots with <null>, which
    // is_valid_box_frame mistakes for a legal box end and would otherwise fabricate a
    // box from second-ranked coordinate candidates — looping forever.
    if (x0[0] == tok.im_end || x0[0] == tok.null_tok) {
        return handle_pattern(x0, tok, mode);
    }

    // prefer the averaged box; else a ref span; else fall back to the greedy block
    std::vector<llama_token> block = decode_bbox_avg(rows, tok, mode);
    if (block.empty()) {
        block = decode_ref(rows, tok);
    }
    const std::vector<llama_token> & new_tokens = block.empty() ? x0 : block;

    return handle_pattern(new_tokens, tok, mode);
}
