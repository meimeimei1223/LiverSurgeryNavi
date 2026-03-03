#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D texture1;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;
uniform bool useTexture;
uniform vec4 vertColor;

void main()
{
    // アルファが0なら早期リターン
    if (vertColor.a < 0.01) {
        discard;
    }
    
    // テクスチャ座標が無効な場合は描画しない
    if (TexCoord.x < 0.0 || TexCoord.y < 0.0) {
        discard;
    }
    
    // ベースカラー
    vec3 objectColor;
    float alpha;
    
    if (useTexture) {
        vec4 texColor = texture(texture1, TexCoord);
        objectColor = texColor.rgb;
        alpha = texColor.a * vertColor.a;
    } else {
        objectColor = vertColor.rgb;
        alpha = vertColor.a;
    }
    
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, alpha);
}
