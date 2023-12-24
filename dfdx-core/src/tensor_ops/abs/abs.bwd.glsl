#version 460 core

#extension GL_ARB_compute_shader: enable
#extension GL_ARB_shader_storage_buffer_object: enable

layout(local_size_x = 128) in;

layout(std430, binding = 1) buffer inpBlock {
    TYPENAME inp[];
};

layout(std430, binding = 2) buffer outpBlock {
    TYPENAME outp[];
};

layout(std430, binding = 3) buffer input_gradBlock {
    TYPENAME input_grad[];
};

layout(std430, binding = 4) buffer output_gradBlock {
    TYPENAME output_grad[];
};

void main() {
    TYPENAME dx = sign(inp[gl_GlobalInvocationID.x]);

    input_grad[gl_GlobalInvocationID.x] = dx * output_grad[gl_GlobalInvocationID.x];
}
