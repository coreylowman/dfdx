#version 460 core

#extension GL_EXT_shader_atomic_float: enable
#extension SPV_EXT_shader_atomic_float_add: enable
#extension GL_ARB_compute_shader: enable
#extension GL_ARB_shader_storage_buffer_object: enable
#extension ARB_shader_atomic_counter_ops: enable
#extension VK_EXT_shader_atomic_float: enable

layout(local_size_x = 128) in;

layout(std430, binding = 1) buffer inpBlock {
    TYPENAME inp[];
};

layout(std430, binding = 2) buffer outpBlock {
    TYPENAME outp[];
};

layout(std430, binding = 3) buffer params {
    uint chunk_len;
    TYPENAME elems_per_thread;
};

layout(std430, binding = 4) buffer dimsBlock {
    uint dims[];
};

layout(std430, binding = 5) buffer stridesBlock {
    uint strides[];
};

uint next_power_of_two(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v++;
    return v;
}

uint get_strided_index(uint idx) {
    uint strided_i = 0;
    for (uint d = 0; d < dims.length(); d++) {
        uint dim_idx = dims.length() - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

void chunk_sum(
    uint chunk_len,
    TYPENAME data
) {
    TYPENAME buf[1024];

    // assumes that threads where i >= numel have already exited
    uint i = gl_GlobalInvocationID.x;
    uint block_i = gl_WorkGroupID.x;

    // Fall back to atomicAdd if chunk_len is small to reduce overhead
    if (chunk_len <= 2) {
        atomicAdd(outp[i / chunk_len], data);
        return;
    }
    buf[block_i] = data;

    uint chunk_i = i % chunk_len;
    uint chunk_start = max(int(block_i - chunk_i), 0);
    uint chunk_end = min(uint(block_i + chunk_len - chunk_i), gl_WorkGroupSize.x);

    chunk_i = block_i - chunk_start;

    uint max_chunk_len = min(chunk_end - chunk_start, gl_WorkGroupSize.x);
    uint incr = next_power_of_two(max_chunk_len) >> 1;

    barrier();

    // Uses sequential addressing as discussed in
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    for (; incr > 0; incr >>= 1) {
        uint block_i_2 = block_i + incr;

        if (block_i_2 < chunk_end && chunk_i < incr) {
            // This is sound because __syncthreads and the conditions above
            // ensure that no data races occur
            buf[block_i] += buf[block_i_2];
        }

        barrier();
    }

    if (block_i == chunk_start) {
        atomicAdd(outp[i / chunk_len], buf[block_i]);
    }
}

void main() {
    if (gl_GlobalInvocationID.x >= inp.length()) {
        return;
    }

    uint inp_idx = get_strided_index(gl_GlobalInvocationID.x);

    chunk_sum(chunk_len, inp[inp_idx] * elems_per_thread);
}
