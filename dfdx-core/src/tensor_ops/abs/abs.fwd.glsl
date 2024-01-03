#version 460 core

#extension GL_ARB_compute_shader: enable
#extension GL_ARB_shader_storage_buffer_object: enable

layout(local_size_x = 128) in;

layout(std430, binding = 1) buffer inpBlock {
    TYPENAME inp[];
};

layout(std430, binding = 2) buffer outpBlock{
    TYPENAME outp[];
};

void main() {
    if (inp.length() == 0) {
        outp[gl_GlobalInvocationID.x] = abs(outp[gl_GlobalInvocationID.x]);
    } else {
        outp[gl_GlobalInvocationID.x] = abs(inp[gl_GlobalInvocationID.x]);
    }
}
