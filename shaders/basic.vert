#version 330 core 

layout (location = 0) in vec3 pos;			
layout (location = 1) in vec3 normal;	
layout (location = 2) in vec2 texCoord;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 clipPlane;  // 追加：クリッピング平面

void main()
{
    FragPos = vec3(model * vec4(pos, 1.0f));
    Normal = mat3(transpose(inverse(model))) * normal;
    TexCoord = texCoord;
    
    // クリッピング距離を計算
    gl_ClipDistance[0] = dot(vec4(FragPos, 1.0), clipPlane);
    
    gl_Position = projection * view * model * vec4(pos, 1.0f);
}
