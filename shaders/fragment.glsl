#version 330

//#define USE_ANTI_ALIASING 0
#define eps 1e-3
#define samples 5
#define max_depth 4
#define step 1e-1
#define max_it 500
#define penumbra_factor 12.0
#define PRIM_NUM 4
#define LIGHTS_NUM 2
#define ambient_light 0.3
#define MAX_ANGLE 0.04

//uniform bool moved = true;

uniform samplerCube Cube;
uniform sampler2D Plane;

in vec2 fragmentTexCoord;

layout(location = 0) out vec4 fragColor;

//uniform bool show_soft_shadows = false;
uniform bool show_fog = false;

uniform int g_screenWidth = 512;
uniform int g_screenHeight = 512;

uniform mat4 g_rayMatrix;

struct Material {
        float ambient;
        float diffuse;
        float spec;
        float reflection;
        float refraction;
        float eta;
        };


uniform Material materials[] = Material[] (
                                Material(1, 1, -1, 0, 0, 1),
                                Material(1, 1, 4096, 0.5, 0, 1),
                                Material(1, 1, -1, 0, 0.7, 1.5),
                                Material(1, 1, -1,0, 0, 1.0));

uniform int objects[] = int[](0, 1, 2, 3, 2);

struct Point_light {
        float intensity;
        vec3 pos;
        float kc;
        float kl;
        float kq;
    };

uniform Point_light lights[] = Point_light[](
                            Point_light(.4, vec3(0, 3, 0), 0.91, 0.01, 0.00001),
                            Point_light(0.4, vec3(1.2, 4, 1.2), 0.91, 0.01, 0.00001));

struct Intersect {
        int id;
        vec3 pos;
        vec3 n;//outer nornal
};

struct Ray {
        vec3 pos;
        vec3 dir;
        };

struct Stack_frame {
        int phase; // 0 - nothing done 1 - reflect was computed 2 - refract was computed
        Ray ray;
        vec4 color;
        Intersect hit;
        int id;
        int material_id;
        int depth;
};

int esp = 1;
float max_dist;

Stack_frame stack[5];

vec3 EyeRayDir(float x, float y, float w, float h) {
	float fov = 3.141592654f/(2.0f); 
    vec3 ray_dir;
  
	ray_dir.x = x+.5 - (w/2.0f);
	ray_dir.y = y+.5 - (h/2.0f);
	ray_dir.z = -(w)/tan(fov/2.0f);
	
  return normalize(ray_dir);
}

mat3 rotate_Y(float t) {
	return mat3(cos(t), 0.0, -sin(t),
                0.0, 1.0, 0.0,
                sin(t), 0.0, cos(t));
}

float smin( float a, float b, float k ) {
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float sdBox( vec3 p, vec3 b ) {
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sdCappedCylinder( vec3 p, vec2 h ) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCone( vec3 p, vec2 c ) {
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}

float sdEllipsoid( in vec3 p, in vec3 r ) {
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}

float sdPawn(vec3 p) {
    float d1 = sdCappedCylinder(p - vec3(0, 0.3, 0), vec2(0.04, 0.1));
    float d2 = max(sdCone((p - vec3(0, 0.3, 0)).xzy, vec2(0.85, 0.5)), -p.y);
    d1 = smin(d1, d2, 0.05);
    d2 = sdTorus(p-vec3(0, 0.39, 0), vec2(0.04, 0.035));
    d1 = smin(d1, d2, 0.05);
    d2 = sdEllipsoid(p - vec3(0, 0.48, 0), vec3(0.05, 0.07, 0.05));
    return min(d1, d2);
}

float sdKing(vec3 p) {
    float d1 = max(sdCone((p - vec3(0, 0.4, 0)).xzy, vec2(0.85, 0.5)), -p.y);
    float d2 = sdCappedCylinder(p - vec3(0, 0.45, 0), vec2(0.05, 0.2));
    d1 = smin(d1, d2, 0.1);
    vec3 q = (p - vec3(0, 0.6, 0)).xzy;
    q.z = -q.z;
    d2 = max(sdCone(q, vec2(0.85, 0.5)), p.y - 0.78);
    d1 = min(d1, d2);//, 0.1);
    d2 = sdTorus(p - vec3(0, 0.6, 0), vec2(0.055, 0.05));
    d1 = smin(d1, d2, 0.05);
    d2 = smin(sdBox(p - vec3(0, 0.83, 0), vec3(0.02, 0.05, 0.02)),
                sdBox(p - vec3(0, 0.83, 0), vec3(0.05, 0.02, 0.02)), 0.02);
    return min(d1, d2);
}


float DistanceEvaluation(vec3 p, int id){
    switch(id){
        case 0:
            return p.y;
        case 1:
            vec3 d = abs(p - vec3(0, 0.08, 0)) - vec3(2.4, 0.08, 2.4);
            return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
        case 2:
            return sdPawn(p - vec3(-0.3, 0.16, 0.3));
        case 3:
        //    p -= vec3(4, 1, -3);
        //    return fractal1(rotate_Y(3.14/6)*p);
        //case 4:
            return sdKing(p - vec3(-0.9, 0.16, -0.3));
    }
    return 0.;
}

vec4 color(vec3 pos, int id) {
    switch(id) {
        case 0:
            return texture(Plane, pos.xz);
        case 1:
            if (pos.y < 0.16 - eps/2) {
                return vec4(1, 0.8, 0.83, 1);
            }
            ivec2 p = ivec2(ceil(pos.xz/0.6));
            if (mod((p.x + p.y), 2) == 0) {
                return vec4(0.96, 0.96, 0.86, 1);
            } else {
                return vec4(0.2, 0.2, 0.2, 1);
            }
        case 2:
        case 4:
            return vec4(1, 1,1, 1);
        case 3:
            return vec4(0.8, 0.4, 0.9, 1);

    }
}

vec3 EstimateNormal(vec3 z, int id) {
    vec3 z1 = z + vec3(eps, 0, 0);
    vec3 z2 = z - vec3(eps, 0, 0);
    vec3 z3 = z + vec3(0, eps, 0);
    vec3 z4 = z - vec3(0, eps, 0);
    vec3 z5 = z + vec3(0, 0, eps);
    vec3 z6 = z - vec3(0, 0, eps);
    float dx = DistanceEvaluation(z1, id) - DistanceEvaluation(z2, id);
    float dy = DistanceEvaluation(z3, id) - DistanceEvaluation(z4, id);
    float dz = DistanceEvaluation(z5, id) - DistanceEvaluation(z6, id);
    return normalize(vec3(dx, dy, dz));
}

float MinDist(vec3 pos, out int object_id) {
    float cur_dist, min_dist = max_dist*10;
    object_id = 0;
    for(int i = 0; i < PRIM_NUM; i++) {
        cur_dist = DistanceEvaluation(pos, i);
        if (cur_dist < min_dist) {
            min_dist = cur_dist;
            object_id = i;
        }
    }
    return min_dist;
}

float Shadow(vec3 point_pos, vec3 ray_dir, float maxt) {
    float ph = 1e20, y, h;
    float res = 1.0;
    int id;
    float loc_eps = eps*0.01;
    float t= 2*eps;
    while (t < maxt)
    {
        float h = MinDist(point_pos + t*ray_dir, id);
        if(h < loc_eps) {
            return 0.;
        }
//        if (show_soft_shadows) {
//            float y = h*h/(2.0*ph);
//            ph = h;
//            float d = sqrt(h*h-y*y);
//            res = min(res, penumbra_factor*h/max(0.0,t-y));
//            }
        t += h;
    }
    return res;
}

Intersect ray_intersection(Ray ray) {
    float cur_dist, min_dist;
    float t = 0;
    int object_id = -1;
    for(int i = 0; i < (max_it); i++) {
        min_dist = MinDist(ray.pos + t*ray.dir, object_id);
        t += min_dist;
        if (abs(min_dist) < eps/5) {
            break;
        }
        if (t > max_dist) {
            object_id = -1;
            t = max_dist;
            break;
        }
    }
    ray.pos += t*ray.dir;
    vec3 n;
    if (object_id != -1) {
        n = EstimateNormal(ray.pos, object_id);
    }
    Intersect res = Intersect(object_id, ray.pos, n);
    return res;
}

float get_occlusion(Intersect hit) {
    float occlusion = 1.0f;
    float dist;
    int id;
    vec3 step_vector = step*hit.n;
    vec3 cur_pos = hit.pos + step*hit.n;
    for (int i = 1; i <= samples; i++) {
        dist = MinDist(cur_pos, id);
        occlusion -= pow(2, samples - i + 2)*(i * step - dist) / i*step;
        cur_pos += step_vector;
    }

    return clamp(occlusion, 0, 1);
}

float get_shade(Intersect hit, vec3 ray_dir) {
    Material material = materials[objects[hit.id]];
    //ambient
    float intensity = ambient_light * material.ambient * get_occlusion(hit);
    float shadow;
    Intersect result;//
    for (int i = 0; i < LIGHTS_NUM; i++) {
        vec3 l = normalize(lights[i].pos - hit.pos);
        //shadow
        shadow = Shadow(hit.pos, l, length(lights[i].pos - hit.pos));
        if (shadow == 0.0) {
            continue;
        }
        float d = length(hit.pos - lights[i].pos);
        float att = lights[i].kc + lights[i].kl*d + lights[i].kq*d*d;
        //diffuse
        float n_dot_l = max(dot(hit.n, l), 0);
        intensity += lights[i].intensity* material.diffuse*n_dot_l*shadow/att;
             //specular
        if (material.spec != -1) {
            float r_dot_v = dot(2*n_dot_l*hit.n - l, -ray_dir);
            if (r_dot_v > 0) {
                intensity += lights[i].intensity*pow(r_dot_v, material.spec)*shadow/att;
            }
        }
    }
    return intensity;
}

vec4 fog(float dist, vec4 color ) {
    return mix(color,vec4(0.4,0.4,0.6, 1), smoothstep(0.0,1.0,sqrt(dist)/5.0) );
}


vec4 ray_march(inout Stack_frame frame, vec4 prev_ret) {
    if (frame.phase == 0) {
        frame.color = vec4(0, 0,0,0);
        frame.hit = ray_intersection(frame.ray);
        if (frame.hit.id == -1) {
            frame.color = texture(Cube, frame.ray.dir);
            esp--;
            if (show_fog)
                return fog(length(frame.hit.pos - frame.ray.pos), frame.color);;
            return frame.color;
        }
        frame.id = frame.hit.id;
        frame.material_id = objects[frame.id];
        if (frame.depth >= max_depth) {
             esp--;
             return color(frame.hit.pos, frame.id)*get_shade(frame.hit, frame.ray.dir);
        }
        frame.color += color(frame.hit.pos, frame.id)*get_shade(frame.hit, frame.ray.dir) * (1 - materials[frame.material_id].reflection
                                                                   - materials[frame.material_id].refraction);
        if (materials[frame.material_id].reflection > 0) {
            vec3 refl = reflect(frame.ray.dir, frame.hit.n);
            vec3 shift = normalize(frame.ray.pos - frame.hit.pos);
            Ray refl_ray = Ray(frame.hit.pos + 2*eps*shift, refl);
            stack[esp] = Stack_frame(0, refl_ray, vec4(0, 0, 0, 0), frame.hit, frame.id, frame.material_id, frame.depth + 1);
            esp++;
            frame.phase = 1;
            return frame.color;
        } else {
            frame.phase = 1;
        }
    }
    if (frame.phase == 1) {
        frame.color += materials[frame.material_id].reflection * prev_ret;
        if (materials[frame.material_id].refraction > 0) {
            float eta1, eta2;
            int id;
            if (MinDist(frame.hit.pos - 3*eps*frame.ray.dir, id) > 0) {
                eta1 = 1;
            } else {
                eta1 = materials[frame.material_id].eta;
            }
            if (MinDist(frame.hit.pos + 3*eps*frame.ray.dir, id) > 0) {
                eta2 = 1;
            } else {
                eta2 = materials[frame.material_id].eta;
            }
            vec3 refr = refract(frame.ray.dir, frame.hit.n, eta1/eta2);
            if (refr == vec3(0, 0, 0)) {
                vec3 refl = reflect(frame.ray.dir, frame.hit.n);
                vec3 shift = normalize(frame.ray.pos - frame.hit.pos);
                Ray refl_ray = Ray(frame.hit.pos + 2*eps*shift, refl);
                stack[esp] = Stack_frame(0, refl_ray, vec4(0, 0, 0, 0), frame.hit, frame.id, frame.material_id, frame.depth + 1);
                esp++;
                frame.phase = 2;
                return frame.color;
                //esp--;
                //return frame.color + color(frame.hit.pos, frame.id) * materials[frame.material_id].refraction * get_shade(frame.hit, frame.ray.dir);
            }
            vec3 shift = normalize(-frame.ray.pos + frame.hit.pos);
            Ray refr_ray = Ray(frame.hit.pos + 2*eps*shift, -refr);
            stack[esp] = Stack_frame(0, refr_ray, vec4(0, 0, 0, 0), frame.hit, frame.id, frame.material_id, frame.depth + 1);
            esp++;
            frame.phase = 2;
            return frame.color;
        } else {
            frame.phase = 2;
        }
    }
    if (frame.phase == 2) {
        frame.color += materials[frame.material_id].refraction*prev_ret;
        frame.phase = 3;
    }
    esp--;
    if (show_fog)
        return fog(length(frame.hit.pos - frame.ray.pos), frame.color);
    return frame.color;
}


vec4 main_function(float w, float h, float x, float y, float x_shift, float y_shift) {
    vec3 ray_pos = vec3(0,0,0);
    vec3 ray_dir = EyeRayDir(x+x_shift,y+y_shift,w,h);
      // transorm ray with matrix
      //
    ray_pos = (g_rayMatrix*vec4(ray_pos,1)).xyz;
    ray_dir = normalize(mat3(g_rayMatrix)*ray_dir);
    max_dist = ray_pos.y/MAX_ANGLE;
    int id;
    ray_dir *= sign(MinDist(ray_pos, id));
    Stack_frame zero_frame;
    zero_frame.phase = 0;
    zero_frame.ray = Ray(ray_pos, ray_dir);
    zero_frame.depth = 0;
    stack[0] = zero_frame;
    vec4 c = vec4(0, 0, 0, 0);
    esp = 1;
    while (esp != 0) {
        c = ray_march(stack[esp - 1], c);
    }
    return c;
}

void main(void)
{
    //vec2 pix_coord = 0.5 + mat2(cos(view_angle), sin(view_angle), -sin(view_angle), cos(view_angle))*(fragmentTexCoord -0.5);
    float w = float(g_screenWidth);
    float h = float(g_screenHeight);
  // get curr pixelcoordinates
  //
    float x = fragmentTexCoord.x*w;
    float y =fragmentTexCoord.y*h;
  // generate initial ray
//  #if USE_ANTI_ALIASING == 1
//    if (moved) {
//        fragColor = main_function(w, h, x, y, 0, 0);
//    } else {
//        fragColor = vec4(0, 0, 0, 0);
//        for (float x_shift = -1./3; x_shift < 0.5; x_shift += 2./3)
//            for (float y_shift = -1./3; y_shift < 0.5; y_shift += 2./3) {
//                fragColor += main_function(w, h, x, y, x_shift, y_shift);
//            }
//        fragColor /= 4;
//    }
//  #else
    fragColor = main_function(w, h, x, y, 0, 0);
//  #endif
}


