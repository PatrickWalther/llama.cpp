#pragma once

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Chunked pipeline over the first T_range tokens only; final state written to state_dst.
// Partial ranges (T_range < n_tokens) require a single sequence.
void ggml_cuda_op_gated_delta_net_chunked_range(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                                int T_range, float * state_dst);
