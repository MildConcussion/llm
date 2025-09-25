import torch


def compile_phi2_kernel(head_dim: int) -> any:
    """Compile a Metal shader implementing the symmetric power feature map (p=2).

    Input x: [num_tokens, head_dim] float32 (MPS)
    Output y: [num_tokens, comb(head_dim+1, 2)] float32 (SPOW basis: squares and pairwise products, i<=j)

    Returns a compiled library exposing `phi2_expand`.
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    // Computes y for each token: y = [x_0*x_0, x_0*x_1, ..., x_0*x_{{d-1}}, x_1*x_1, x_1*x_2, ..., x_{{d-1}}*x_{{d-1}}]
    // Where d = head_dim, total out_dim = d*(d+1)/2, layout is row-major per token
    kernel void phi2_expand(
        device const float* x [[buffer(0)]],     // [num_tokens, head_dim]
        device float* y [[buffer(1)]],           // [num_tokens, out_dim]
        constant int& num_tokens [[buffer(2)]],
        constant int& head_dim_c [[buffer(3)]],
        constant int& out_dim [[buffer(4)]],
        uint idx [[thread_position_in_grid]])
    {
        int token_idx = int(idx);
        if (token_idx >= num_tokens) return;

        int d = head_dim_c;
        const device float* x_row = x + token_idx * d;
        device float* y_row = y + token_idx * out_dim;

        // Compute upper-triangular products (i<=j) in a single pass
        int out_offset = 0;
        for (int i = 0; i < d; ++i) {
            float xi = x_row[i];
            // Diagonal term
            y_row[out_offset++] = xi * xi;
            // Off-diagonals
            for (int j = i + 1; j < d; ++j) {
                y_row[out_offset++] = xi * x_row[j];
            }
        }
    }
    """

    return torch.mps.compile_shader(kernel_source)


def phi2_out_dim(head_dim: int) -> int:
    return head_dim * (head_dim + 1) // 2

def compile_phi2_kernel_normalized(head_dim: int) -> any:
    """Compile a Metal shader for normalized p=2 SPOW with per-output normalization factors.

    Expects additional buffer norm_factors[length=out_dim].
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void phi2_expand_norm(
        device const float* x [[buffer(0)]],     // [num_tokens, head_dim]
        device float* y [[buffer(1)]],           // [num_tokens, out_dim]
        device const float* norm_factors [[buffer(2)]], // [out_dim]
        constant int& num_tokens [[buffer(3)]],
        constant int& head_dim_c [[buffer(4)]],
        constant int& out_dim [[buffer(5)]],
        uint idx [[thread_position_in_grid]])
    {
        int token_idx = int(idx);
        if (token_idx >= num_tokens) return;

        int d = head_dim_c;
        const device float* x_row = x + token_idx * d;
        device float* y_row = y + token_idx * out_dim;

        int out_offset = 0;
        for (int i = 0; i < d; ++i) {
            float xi = x_row[i];
            y_row[out_offset] = norm_factors[out_offset] * (xi * xi);
            out_offset++;
            for (int j = i + 1; j < d; ++j) {
                y_row[out_offset] = norm_factors[out_offset] * (xi * x_row[j]);
                out_offset++;
            }
        }
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_update_state_phi2() -> any:
    """Compile a Metal shader that updates (C,D) state and (C) norm_state from a chunk of (T,C) and (T,D).

    Launch with 2D grid [C, D].
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void update_state_phi2(
        device const float* phi_k [[buffer(0)]],  // [chunk_size, C]
        device const float* v [[buffer(1)]],      // [chunk_size, D]
        device float* state [[buffer(2)]],        // [C, D]
        device float* norm_state [[buffer(3)]],   // [C]
        constant float& decay [[buffer(4)]],
        constant int& chunk_size [[buffer(5)]],
        constant int& c_dim [[buffer(6)]],
        constant int& d_dim [[buffer(7)]],
        uint2 tid [[thread_position_in_grid]])
    {
        int c_idx = int(tid.x);
        int d_idx = int(tid.y);
        if (c_idx >= c_dim || d_idx >= d_dim) return;

        // Accumulate contribution from the chunk
        float sum = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            sum += phi_k[t * c_dim + c_idx] * v[t * d_dim + d_idx];
        }

        // Update state with decay
        state[c_idx * d_dim + d_idx] = decay * state[c_idx * d_dim + d_idx] + sum;

        // Update normalization for d_idx == 0 threadline only
        if (d_idx == 0) {
            float norm_sum = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                norm_sum += phi_k[t * c_dim + c_idx];
            }
            norm_state[c_idx] = decay * norm_state[c_idx] + norm_sum;
        }
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_update_state_phi2_bh() -> any:
    """Compile a Metal shader that updates batched (BH,C,D) state and (BH,C) norm_state.

    1D grid over (BH*C); loops over D inside the kernel for stability.
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void update_state_phi2_bh(
        device const float* phi_k [[buffer(0)]],  // [BH, L, C]
        device const float* v [[buffer(1)]],      // [BH, L, D]
        device float* state [[buffer(2)]],        // [BH, C, D]
        device float* norm_state [[buffer(3)]],   // [BH, C]
        device const float* decay_per_bh [[buffer(4)]], // [BH]
        constant int& BH [[buffer(5)]],
        constant int& L [[buffer(6)]],
        constant int& Cdim [[buffer(7)]],
        constant int& Ddim [[buffer(8)]],
        uint idx [[thread_position_in_grid]])
    {
        int total = BH * Cdim;
        int flat = int(idx);
        if (flat >= total) return;
        int bh = flat / Cdim;
        int c_idx = flat % Cdim;

        // Base offsets
        int phi_bh_off = bh * L * Cdim;
        int v_bh_off = bh * L * Ddim;
        int state_bh_off = bh * Cdim * Ddim;
        int norm_bh_off = bh * Cdim;

        float decay = decay_per_bh[bh];
        // Update all D for this (bh,c)
        for (int d_idx = 0; d_idx < Ddim; ++d_idx) {
            float sum = 0.0f;
            for (int t = 0; t < L; ++t) {
                sum += phi_k[phi_bh_off + t * Cdim + c_idx] * v[v_bh_off + t * Ddim + d_idx];
            }
            int st_idx = state_bh_off + c_idx * Ddim + d_idx;
            state[st_idx] = decay * state[st_idx] + sum;
        }

        float norm_sum = 0.0f;
        for (int t = 0; t < L; ++t) {
            norm_sum += phi_k[phi_bh_off + t * Cdim + c_idx];
        }
        int ns_idx = norm_bh_off + c_idx;
        norm_state[ns_idx] = decay * norm_state[ns_idx] + norm_sum;
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_ts_pow_intra() -> any:
    """Compile a Metal kernel that computes intra-chunk outputs and norms with causal masking.

    1D grid over (BH*L); loops over D inside the kernel for stability.
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void ts_pow_intra(
        device const float* phi_q [[buffer(0)]],  // [BH, L, C]
        device const float* phi_k [[buffer(1)]],  // [BH, L, C]
        device const float* v [[buffer(2)]],      // [BH, L, D]
        device float* out [[buffer(3)]],          // [BH, L, D]
        device float* norm [[buffer(4)]],         // [BH, L]
        constant int& BH [[buffer(5)]],
        constant int& L [[buffer(6)]],
        constant int& Cdim [[buffer(7)]],
        constant int& Ddim [[buffer(8)]],
        uint idx [[thread_position_in_grid]])
    {
        int total = BH * L;
        int flat = int(idx);
        if (flat >= total) return;
        int bh = flat / L;
        int i = flat % L;

        int phi_bh_off = bh * L * Cdim;
        int v_bh_off = bh * L * Ddim;
        int out_bh_off = bh * L * Ddim;
        int norm_bh_off = bh * L;

        float acc_norm = 0.0f;
        // Precompute dot products for this (bh,i) against all j<=i
        // Accumulate norms and per-d output
        // To keep cache-friendly, compute per-d inside loop over j
        // Initialize output row to 0
        for (int d = 0; d < Ddim; ++d) {
            out[out_bh_off + i * Ddim + d] = 0.0f;
        }

        for (int j = 0; j <= i; ++j) {
            float dot = 0.0f;
            int q_off = phi_bh_off + i * Cdim;
            int k_off = phi_bh_off + j * Cdim;
            for (int c = 0; c < Cdim; ++c) {
                dot += phi_q[q_off + c] * phi_k[k_off + c];
            }
            acc_norm += dot;
            int v_row = v_bh_off + j * Ddim;
            int out_row = out_bh_off + i * Ddim;
            for (int d = 0; d < Ddim; ++d) {
                out[out_row + d] += dot * v[v_row + d];
            }
        }

        norm[norm_bh_off + i] = acc_norm;
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_query_state_phi2() -> any:
    """Compile a Metal kernel that computes inter-chunk query x [state,norm_state].

    Computes, for each (bh, i):
      out[bh, i, d]   = sum_c phi_q[bh, i, c] * state[bh, c, d]
      out_norm[bh, i] = sum_c phi_q[bh, i, c] * norm_state[bh, c]

    1D grid over (BH*L); loops C and D inside the kernel for stability and to avoid atomics.
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void query_state_phi2(
        device const float* phi_q [[buffer(0)]],   // [BH, L, C]
        device const float* state  [[buffer(1)]],  // [BH, C, D]
        device const float* norm_state [[buffer(2)]], // [BH, C]
        device float* out [[buffer(3)]],           // [BH, L, D]
        device float* out_norm [[buffer(4)]],      // [BH, L]
        constant int& BH [[buffer(5)]],
        constant int& L  [[buffer(6)]],
        constant int& Cdim [[buffer(7)]],
        constant int& Ddim [[buffer(8)]],
        uint idx [[thread_position_in_grid]])
    {
        int total = BH * L;
        int flat = int(idx);
        if (flat >= total) return;

        int bh = flat / L;
        int i  = flat % L;

        // Base offsets
        int phi_bh_off   = bh * L * Cdim;
        int state_bh_off = bh * Cdim * Ddim;
        int norm_bh_off  = bh * Cdim;
        int out_bh_off   = bh * L * Ddim;

        // Compute out_norm first (dot with norm_state)
        float acc_norm = 0.0f;
        int q_off = phi_bh_off + i * Cdim;
        for (int c = 0; c < Cdim; ++c) {
            acc_norm += phi_q[q_off + c] * norm_state[norm_bh_off + c];
        }
        out_norm[bh * L + i] = acc_norm;

        // Compute each D column as dot over C
        int out_row = out_bh_off + i * Ddim;
        for (int d = 0; d < Ddim; ++d) {
            float acc = 0.0f;
            int state_col_off = d; // state index = state_bh_off + c*Ddim + d
            for (int c = 0; c < Cdim; ++c) {
                float qv = phi_q[q_off + c];
                float sv = state[state_bh_off + c * Ddim + d];
                acc += qv * sv;
            }
            out[out_row + d] = acc;
        }
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_qk_pow2_intra_fused() -> any:
    """Compile a Metal kernel that fuses φ2 expansion and intra-chunk accumulation.

    Inputs:
      q: [BH, L, D]
      k: [BH, L, D]
      v: [BH, L, Dv]

    Computes:
      - φ2(q_i) · φ2(k_j) for j<=i using implicit expansion and identity:
        (q_i^T k_j)^2 = (sum_d q_i[d]*k_j[d])^2
      - Accumulate out[i,:] += (q_i^T k_j)^2 * v[j,:]
      - norm[i] += (q_i^T k_j)^2

    Strategy:
      - Threads over (BH*L)
      - Inner loop j over 0..i, computing dot(q_i, k_j) and squaring it
      - Loop D in scalar, with optional float4 vectorization when D%4==0
      - Loop Dv for output accumulation
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void qk_pow2_intra_fused(
        device const float* q [[buffer(0)]],   // [BH, L, D]
        device const float* k [[buffer(1)]],   // [BH, L, D]
        device const float* v [[buffer(2)]],   // [BH, L, Dv]
        device float* out [[buffer(3)]],       // [BH, L, Dv]
        device float* norm [[buffer(4)]],      // [BH, L]
        constant int& BH [[buffer(5)]],
        constant int& L  [[buffer(6)]],
        constant int& D  [[buffer(7)]],
        constant int& Dv [[buffer(8)]],
        uint idx [[thread_position_in_grid]])
    {
        int total = BH * L;
        int flat = int(idx);
        if (flat >= total) return;
        int bh = flat / L;
        int i  = flat % L;

        int q_bh_off = bh * L * D;
        int k_bh_off = bh * L * D;
        int v_bh_off = bh * L * Dv;
        int out_bh_off = bh * L * Dv;
        int norm_bh_off = bh * L;

        // Zero output row
        for (int dv = 0; dv < Dv; ++dv) {
            out[out_bh_off + i * Dv + dv] = 0.0f;
        }

        float acc_norm = 0.0f;

        // Loop over causal positions j <= i
        for (int j = 0; j <= i; ++j) {
            // Compute dot(q[i], k[j])
            float dot = 0.0f;
            int q_off = q_bh_off + i * D;
            int k_off = k_bh_off + j * D;

            // Vectorized path when D%4==0
            if ((D & 3) == 0) {
                int D4 = D >> 2;
                const device float4* q4 = reinterpret_cast<const device float4*>(q + q_off);
                const device float4* k4 = reinterpret_cast<const device float4*>(k + k_off);
                for (int t = 0; t < D4; ++t) {
                    float4 a = q4[t];
                    float4 b = k4[t];
                    float4 prod = a * b;
                    dot += (prod.x + prod.y + prod.z + prod.w);
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    dot += q[q_off + d] * k[k_off + d];
                }
            }

            float w = dot * dot;
            acc_norm += w;

            int v_row = v_bh_off + j * Dv;
            int out_row = out_bh_off + i * Dv;
            // Accumulate output with weight w
            for (int dv = 0; dv < Dv; ++dv) {
                out[out_row + dv] += w * v[v_row + dv];
            }
        }

        norm[norm_bh_off + i] = acc_norm;
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_discumsum_state() -> any:
    """Compile a Metal kernel that performs discounted cumulative sum over chunked states.

    Inputs:
      - state_in:     [N, BH, C, D]
      - norm_in:      [N, BH, C]
      - lambda_bh:    [N, BH] (per-chunk discount per BH) or broadcastable scalar in host call

    Outputs (written in-place to separate output buffers):
      - state_out:    [N, BH, C, D]
      - norm_out:     [N, BH, C]

    Semantics per chunk index i (causal prefix scan):
      state_out[i] = lambda_bh[i] * state_out[i-1] + state_in[i]
      norm_out[i]  = lambda_bh[i] * norm_out[i-1]  + norm_in[i]

    Grid:
      - 1D over (BH*C) and loop over D; outer for over N in the kernel for better cache locality.
    """
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void discumsum_state(
        device const float* state_in   [[buffer(0)]], // [N, BH, C, D]
        device const float* norm_in    [[buffer(1)]], // [N, BH, C]
        device const float* lambda_bh  [[buffer(2)]], // [N, BH]
        device float* state_out        [[buffer(3)]], // [N, BH, C, D]
        device float* norm_out         [[buffer(4)]], // [N, BH, C]
        constant int& N     [[buffer(5)]],
        constant int& BH    [[buffer(6)]],
        constant int& Cdim  [[buffer(7)]],
        constant int& Ddim  [[buffer(8)]],
        uint idx [[thread_position_in_grid]])
    {
        int total = BH * Cdim;
        int flat = int(idx);
        if (flat >= total) return;
        int bh = flat / Cdim;
        int c  = flat % Cdim;

        // Base strides
        int state_stride_bhcd = Cdim * Ddim;
        int state_stride_n    = BH * state_stride_bhcd;
        int norm_stride_bhc   = Cdim;
        int norm_stride_n     = BH * norm_stride_bhc;

        // Initialize prefix at chunk 0
        // Copy first chunk directly (no discount on i=0)
        int s0_off = 0 * state_stride_n + bh * state_stride_bhcd + c * Ddim;
        int so0_off = s0_off;
        for (int d = 0; d < Ddim; ++d) {
            state_out[so0_off + d] = state_in[s0_off + d];
        }
        int n0_off = 0 * norm_stride_n + bh * norm_stride_bhc + c;
        norm_out[n0_off] = norm_in[n0_off];

        // Prefix scan over remaining chunks
        for (int i = 1; i < N; ++i) {
            int lam_off = i * BH + bh;
            float lam = lambda_bh[lam_off];

            int s_in_off  = i * state_stride_n + bh * state_stride_bhcd + c * Ddim;
            int s_out_off = s_in_off;
            int s_prev_off = (i - 1) * state_stride_n + bh * state_stride_bhcd + c * Ddim;

            for (int d = 0; d < Ddim; ++d) {
                float prev = state_out[s_prev_off + d];
                float cur  = state_in[s_in_off + d];
                state_out[s_out_off + d] = lam * prev + cur;
            }

            int n_in_off  = i * norm_stride_n + bh * norm_stride_bhc + c;
            int n_out_off = n_in_off;
            int n_prev_off = (i - 1) * norm_stride_n + bh * norm_stride_bhc + c;
            float prevn = norm_out[n_prev_off];
            float curn  = norm_in[n_in_off];
            norm_out[n_out_off] = lam * prevn + curn;
        }
    }
    """

    return torch.mps.compile_shader(kernel_source)


def compile_fused_update_state_phi2_bh(head_dim: int, d_tile: int = 32) -> any:
    kernel_source = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant int D_CONST = {head_dim};
    constant int TILE = {d_tile};

    kernel void fused_update_state_phi2_bh(
        device const float* k [[buffer(0)]],
        device const float* v [[buffer(1)]],
        device float* state [[buffer(2)]],
        device float* norm_state [[buffer(3)]],
        device const float* decay [[buffer(4)]],
        constant int& BH [[buffer(5)]],
        constant int& L [[buffer(6)]],
        constant int& D [[buffer(7)]],
        constant int& C [[buffer(8)]],
        uint idx [[thread_position_in_grid]]
    ) {{
        if (idx >= BH * C) return;
        uint bh = idx / C;
        uint global_c = idx % C;
        float dec = decay[bh];

        float acc_norm = 0.0f;
        float acc[D_CONST];
        for (int dd = 0; dd < D_CONST; dd++) acc[dd] = 0.0f;

        int num_tiles = (D + TILE - 1) / TILE;
        int tile_c_off = 0;

        for (int tile = 0; tile < num_tiles; tile++) {{
            int tile_start = tile * TILE;
            int tile_end = min(tile_start + TILE, D);
            int tile_size = tile_end - tile_start;

            int local_c = global_c - tile_c_off;
            if (local_c >= 0 && local_c < tile_size * (tile_size + 1) / 2) {{
                int row = 0;
                int col = local_c;
                while (col >= tile_size - row) {{
                    col -= (tile_size - row);
                    row++;
                }}
                int i = tile_start + row;
                int j = tile_start + row + col;
                float f = (i == j) ? 1.0f : sqrt(2.0f);

                for (int t = 0; t < L; t++) {{
                    uint k_off = bh * L * D + t * D;
                    float ki = k[k_off + i];
                    float kj = k[k_off + j];
                    float prod = ki * kj * f;
                    acc_norm += prod;
                    uint v_off = bh * L * D + t * D;
                    for (int dd = 0; dd < D_CONST; dd++) {{
                        acc[dd] += prod * v[v_off + dd];
                    }}
                }}
            }}
            tile_c_off += tile_size * (tile_size + 1) / 2;
        }}

        norm_state[bh * C + global_c] = dec * norm_state[bh * C + global_c] + acc_norm;

        uint st_base = bh * C * D + global_c * D;
        for (int dd = 0; dd < D_CONST; dd++) {{
            state[st_base + dd] = dec * state[st_base + dd] + acc[dd];
        }}
    }}
    """
    return torch.mps.compile_shader(kernel_source)


def compile_fused_query_state_phi2_bh(head_dim: int, d_tile: int = 32) -> any:
    kernel_source = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant int D_CONST = {head_dim};
    constant int TILE = {d_tile};

    kernel void fused_query_state_phi2_bh(
        device const float* q [[buffer(0)]],
        device const float* state [[buffer(1)]],
        device const float* norm_state [[buffer(2)]],
        device float* out [[buffer(3)]],
        device float* out_norm [[buffer(4)]],
        constant int& BH [[buffer(5)]],
        constant int& L [[buffer(6)]],
        constant int& D [[buffer(7)]],
        constant int& C [[buffer(8)]],
        uint idx [[thread_position_in_grid]]
    ) {{
        if (idx >= BH * L) return;
        uint bh = idx / L;
        uint l = idx % L;

        float acc[D_CONST];
        for (int dd = 0; dd < D_CONST; dd++) acc[dd] = 0.0f;
        float acc_norm = 0.0f;

        int num_tiles = (D + TILE - 1) / TILE;
        int tile_c_off = 0;

        for (int tile = 0; tile < num_tiles; tile++) {{
            int tile_start = tile * TILE;
            int tile_end = min(tile_start + TILE, D);
            int tile_size = tile_end - tile_start;

            for (int row = 0; row < tile_size; row++) {{
                for (int col = row; col < tile_size; col++) {{
                    int i = tile_start + row;
                    int j = tile_start + col;
                    float f = (row == col) ? 1.0f : sqrt(2.0f);

                    float qi = q[bh * L * D + l * D + i];
                    float qj = q[bh * L * D + l * D + j];
                    float prod = qi * qj * f;

                    int cc = tile_c_off + row * tile_size - row * (row - 1) / 2 + (col - row);
                    acc_norm += prod * norm_state[bh * C + cc];

                    for (int dd = 0; dd < D_CONST; dd++) {{
                        acc[dd] += prod * state[bh * C * D + cc * D + dd];
                    }}
                }}
            }}
            tile_c_off += tile_size * (tile_size + 1) / 2;
        }}

        for (int dd = 0; dd < D_CONST; dd++) {{
            out[bh * L * D + l * D + dd] = acc[dd];
        }}
        out_norm[bh * L + l] = acc_norm;
    }}
    """
    return torch.mps.compile_shader(kernel_source)


def compile_intra_inter_fused_chunk_bh(head_dim: int, d_tile: int = 32) -> any:
    kernel_source = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant int D_CONST = {head_dim};
    constant int TILE = {d_tile};

    kernel void intra_inter_fused_chunk_bh(
        device const float* q [[buffer(0)]],
        device const float* k [[buffer(1)]],
        device const float* v [[buffer(2)]],
        device const float* state [[buffer(3)]],
        device const float* norm_state [[buffer(4)]],
        device float* out [[buffer(5)]],
        constant int& BH [[buffer(6)]],
        constant int& L [[buffer(7)]],
        constant int& D [[buffer(8)]],
        constant int& C [[buffer(9)]],
        uint idx [[thread_position_in_grid]]
    ) {{
        if (idx >= BH * L) return;
        uint bh = idx / L;
        uint l = idx % L;

        // Intra-chunk computation
        float acc_local[D_CONST];
        float acc_norm_local = 0.0f;
        for (int dv = 0; dv < D_CONST; dv++) acc_local[dv] = 0.0f;

        for (int j = 0; j <= l; j++) {{
            float dot = 0.0f;
            for (int d = 0; d < D_CONST; d++) {{
                dot += q[bh * L * D + l * D + d] * k[bh * L * D + j * D + d];
            }}
            float w = dot * dot;
            acc_norm_local += w;
            for (int dv = 0; dv < D_CONST; dv++) {{
                acc_local[dv] += w * v[bh * L * D + j * D + dv];
            }}
        }}

        // Inter-chunk computation
        float acc_norm_inter = 0.0f;
        float acc_inter[D_CONST];
        for (int dv = 0; dv < D_CONST; dv++) acc_inter[dv] = 0.0f;

        int num_tiles = (D + TILE - 1) / TILE;
        int tile_c_off = 0;

        for (int tile = 0; tile < num_tiles; tile++) {{
            int tile_start = tile * TILE;
            int tile_end = min(tile_start + TILE, D);
            int tile_size = tile_end - tile_start;

            for (int row = 0; row < tile_size; row++) {{
                for (int col = row; col < tile_size; col++) {{
                    int i = tile_start + row;
                    int j = tile_start + col;
                    float f = (row == col) ? 1.0f : sqrt(2.0f);

                    float qi = q[bh * L * D + l * D + i];
                    float qj = q[bh * L * D + l * D + j];
                    float prod = qi * qj * f;

                    int cc = tile_c_off + row * tile_size - row * (row - 1) / 2 + (col - row);
                    acc_norm_inter += prod * norm_state[bh * C + cc];

                    for (int dd = 0; dd < D_CONST; dd++) {{
                        acc_inter[dd] += prod * state[bh * C * D + cc * D + dd];
                    }}
                }}
            }}
            tile_c_off += tile_size * (tile_size + 1) / 2;
        }}

        // Combine and output
        float denom = acc_norm_local + acc_norm_inter + 1e-6f;
        for (int dv = 0; dv < D_CONST; dv++) {{
            out[bh * L * D + l * D + dv] = (acc_local[dv] + acc_inter[dv]) / denom;
        }}
    }}
    """
    return torch.mps.compile_shader(kernel_source)
