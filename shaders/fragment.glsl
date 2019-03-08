#version 330

#define float2 vec2
#define float3 vec3
#define float4 vec4
#define float4x4 mat4
#define float3x3 mat3
#define eps 1e-4
#define samples 6
#define max_depth 3
#define step 5e-2
#define max_dist 50
#define max_it 1000
#define penumbra_factor 12.0
#define PRIM_NUM 3
#define LIGHTS_NUM 2
#define MANDELBULB_POW 8
#define MANDELBULB_P 2
#define MANDELBULB_Q 2

uniform float ambient = 0.5;

uniform samplerCube Cube;
uniform sampler2D Plane;

in float2 fragmentTexCoord;

layout(location = 0) out vec4 fragColor;

uniform bool show_soft_shadows = false;
uniform bool show_fog = false;

uniform int g_screenWidth = 512;
uniform int g_screenHeight = 512;

uniform float3 g_bBoxMin   = float3(-1,-1,-1);
uniform float3 g_bBoxMax   = float3(+1,+1,+1);

uniform float4x4 g_rayMatrix;

uniform float4 g_bgColor = float4(0,0,0.3,1);

struct Material {
        float ambient;
        float diffuse;
        float spec;
        float reflection;
        float refraction;
        float eta;
        };

struct Object {
        int material_id;
        float4 color;
        };

uniform Material materials[] = Material[] (
                                Material(1, 1, 1024, 0.3, 0, 1),
                                Material(1, 1, 128, 0.3, 0, 1),
                                Material(1, 1, 128, 0, 0.6, 1.2),
                                Material(1, 1, 2048,0.5, 0, 1.0));

uniform Object objects[] = Object[](
//                    Object(1, float3(0, 0, 0), float3(0.6, 0.3, 0),
//                                Material(float4(1, 0.5, 0.5, 1), 1, 1,100, 0, 0, 1)),
//                    Object(2, float3(0, 0.3, 0), float3(0.5, 0.1, 0),
//                                Material(float4(0, 1, 0, 1), 1, 1,100, 0, 0, 1)),
                                Object(0, float4(1, 0, 0, 1)),
                                Object(1, float4(0, 1, 0, 1)),
                                Object(2, float4(0, 0, 1, 1)),
                                Object(3, float4(1, 1, 1, 1)));

struct Point_light {
        float intensity;
        float3 pos;
        float kc;
        float kl;
        float kq;
    };

uniform Point_light lights[] = Point_light[](
                            Point_light(.4, float3(0, 5, 0), 0.1, 0.01, 0.00001),
                            Point_light(0.4, float3(0, 4, 0), 0.1, 0.01, 0.00001));//,
                            //point_light(0.2, float3(0, -2, 2)),
                            //point_light(0.2, float3(0, 2, 0)));

struct Intersect {
        int id;
        float3 pos;
        float3 n;//outer nornal
};

struct Ray {
        float3 pos;
        float3 dir;
        };

struct Stack_frame {
        int phase; // 0 - nothing done 1 - reflect was computed 2 - refract was computed
        Ray ray;
        float4 color;
        Intersect hit;
        int id;
        int material_id;
        int depth;
};

int esp = 1;

Stack_frame stack[4];

float3 EyeRayDir(float x, float y, float w, float h, vec2 shift)
{
	float fov = 3.141592654f/(2.0f); 
    float3 ray_dir;
  
	ray_dir.x = x+shift[0] - (w/2.0f);
	ray_dir.y = y+shift[1] - (h/2.0f);
	ray_dir.z = -(w)/tan(fov/2.0f);
	
  return normalize(ray_dir);
}

mat3 rotate_X(float t)
{
	return mat3(1.0, 0.0, 0.0,
                0.0, cos(t), -sin(t),
                0.0, sin(t), cos(t));
}

mat4 rotate_Y(float t)
{
	return mat4(cos(t), 0.0, -sin(t), 0.0,
                0.0, 1.0, 0.0, 0.0,
                sin(t), 0.0, cos(t), 0,
                0,0,0,1);
}

mat4 rotate_Z(float t)
{
    return mat4(cos(t), -sin(t), 0.0, 0,
                sin(t), cos(t), 0.0, 0,
                0.0, 0.0, 1.0, 0,
                0,0,0,1);
}
bool RayBoxIntersection(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax, inout float tmin, inout float tmax)
{
  ray_dir.x = 1.0f/ray_dir.x;
  ray_dir.y = 1.0f/ray_dir.y;
  ray_dir.z = 1.0f/ray_dir.z;

  float lo = ray_dir.x*(boxMin.x - ray_pos.x);
  float hi = ray_dir.x*(boxMax.x - ray_pos.x);
  
  tmin = min(lo, hi);
  tmax = max(lo, hi);

  float lo1 = ray_dir.y*(boxMin.y - ray_pos.y);
  float hi1 = ray_dir.y*(boxMax.y - ray_pos.y);

  tmin = max(tmin, min(lo1, hi1));
  tmax = min(tmax, max(lo1, hi1));

  float lo2 = ray_dir.z*(boxMin.z - ray_pos.z);
  float hi2 = ray_dir.z*(boxMax.z - ray_pos.z);

  tmin = max(tmin, min(lo2, hi2));
  tmax = min(tmax, max(lo2, hi2));
  
  return (tmin <= tmax) && (tmax > 0.f);
}


vec3 RayMarchConstantFog(float tmin, float tmax, inout float alpha)
{
  float dt = 0.05f;
	float t  = tmin;
	
	alpha  = 1.0f;
	float3 color = float3(0,0,0);
	
	while(t < tmax && alpha > 0.01f)
	{
	  float a = 0.05f;
	  color += a*alpha*float3(1.0f,1.0f,0.0f);
	  alpha *= (1.0f-a);
	  t += dt;
	}
	
	return color;
}

float map(vec3 p) {
    vec3 w = p;
    vec4 trap = vec4(abs(w),dot(w,w));
	float dz = 1.0;
    float r = length(w);
	for( int i=0; i<4; i++ )
    {
            if( r > max_dist)
                    break;
            dz = 8.0*pow(r, 7)*dz + 1.0;
            float b = 8.0*acos( w.y/r);
            float a = 8.0*atan( w.x, w.z );
            w = p + pow(r,8.0) * vec3( sin(b)*sin(a), sin(b)*cos(a), cos(b));
            trap = min( trap, vec4(abs(w),dot(w,w)) );
            r = length(w);

    }
    return 0.25*log(r)*r/dz;
}

vec4 C(vec3 p) {
        vec3 w = p;
        float m = dot(w,w);

        vec4 trap = vec4(abs(w),m);
    	float dz = 1.0;
    	for( int i=0; i<4; i++ ) {
            dz = 8.0*pow(sqrt(m),7.0)*dz + 1.0;
    		//dz = 8.0*pow(m,3.5)*dz + 1.0;
            float r = length(w);
            float b = 8.0*acos(w.y/r);
            float a = 8.0*atan(w.x, w.z );
            w = p + pow(r,8.0) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );
    	trap = min( trap, vec4(abs(w),m) );

                m = dot(w,w);
        		if( m > 1.0 )
                    break;
            }

        vec4 tra = vec4(m,trap.yzw);
        vec3 col = vec3(0.01);
        col = mix( col, vec3(0.10,0.20,0.30), clamp(tra.y,0.0,1.0) );
        col = mix( col, vec3(0.02,0.10,0.30), clamp(tra.z*tra.z,0.0,1.0) );
        col = mix( col, vec3(0.30,0.10,0.02), clamp(pow(tra.w,6.0),0.0,1.0) );
        return vec4(col, 1)*1.5;
}

float sdSphere(vec3 p, float s)
{
  return length(p)-s;
}

float sdTorus(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float opDisplace( vec3 p )
{
    float d1 = sdSphere(p, 1);
    float d2 = sin(10*p.x);
    return d1+d2;
}
float opRep(vec3 p, vec3 c )
{
    vec3 q = mod(p,c)-0.5*c;
    return opDisplace(q);
}

float sdCappedCylinder( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCone( vec3 p, vec2 c )
{
    // c must be normalized
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}

float sdEllipsoid( in vec3 p, in vec3 r )
{
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}

float DistanceEvaluation(vec3 p, int id){
    //Object obj = objects[id];
    switch(id){
        case 0:
            return p.y;
        case 1:
            vec3 d = abs(p - vec3(0, 0.08, 0)) - vec3(2.4, 0.08, 2.4);
            return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
        case 2:
            float d1 = sdTorus(p-vec3(0.3, 0.55, 0.3), vec2(0.06, 0.035));
            float d2 = max(sdCone((p - vec3(0.3, 0.7, 0.3)).xzy, vec2(0.95, 0.3)), p.y - 1.6);
            d1 = min(d1, d2);
            d2 = sdCappedCylinder(p - vec3(0.3, 0.21, 0.3), vec2(0.2, 0.05));
            d1 = min(d1, d2);
            d2 = sdEllipsoid(p - vec3(0.3, 0.65, 0.3), vec3(0.045, 0.07, 0.045));
            return min(d1, d2);
//            //vec2 t = vec2(0.2, 0.04);
//            //p -= vec3(0.3, 0.2, 0.3);
//            //vec2 q = vec2(length(p.xz)-t.x,p.y);
//                        //return length(q)-t.y;
//            p -= vec3(0.3, 0.8, 0.3);
//            mat3 m = rotate_X(-3.14/2);
//            vec3 p1 = m * p;
//            vec2 c = vec2(0.8, 0.3);
//                // c must be normalized
//                float q = length(p1.xy);
//                return max(dot(c,vec2(q,p1.z)), p.y - 1.6);

        case 3: //box
            return sdBox(p - float3(0, -0.5, 0), float3(100, 0.1, 100));
        //case 5:
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
            return vec4(0, 0,1, 1);
}
}

vec3 EstimateNormal(float3 z, int id)
{
    float3 z1 = z + float3(eps, 0, 0);
    float3 z2 = z - float3(eps, 0, 0);
    float3 z3 = z + float3(0, eps, 0);
    float3 z4 = z - float3(0, eps, 0);
    float3 z5 = z + float3(0, 0, eps);
    float3 z6 = z - float3(0, 0, eps);
    float dx = DistanceEvaluation(z1, id) - DistanceEvaluation(z2, id);
    float dy = DistanceEvaluation(z3, id) - DistanceEvaluation(z4, id);
    float dz = DistanceEvaluation(z5, id) - DistanceEvaluation(z6, id);
    return normalize(float3(dx, dy, dz));
}

float MinDist(float3 pos, out int object_id) {
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

float Shadow(float3 point_pos, float3 ray_dir, float maxt)
{
    float ph = 1e20;
    float res = 1.0;
    int id;
    float loc_eps = eps*0.01;
    for (float t = eps*2; t < maxt; )
    {
        float h = MinDist(point_pos + t*ray_dir, id);
        if(h < loc_eps) {
            return 0.;
        }
        if (show_soft_shadows) {
            float y = h*h/(2.0*ph);
            float d = sqrt(h*h-y*y);
            res = min(res, penumbra_factor*h/max(0.0,t-y));
            ph = h;
            }
        t += h;
    }
    return res;
}

Intersect ray_intersection(Ray ray) {
    float cur_dist, min_dist;
    float t = 0;
    int object_id = -1;
    for(int i = 0; i < 100; i++) {
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
    float3 step_vector = step*hit.n;
    float3 cur_pos = hit.pos + step*hit.n;
    for (int i = 1; i <= samples; i++) {
        dist = MinDist(cur_pos, id);
        occlusion -= pow(2, samples - i)*(i * step - dist) / i*step;
        cur_pos += step_vector;
    }

    return max(occlusion, 0);
}

float get_shade(Intersect hit, float3 ray_dir) {
    Object obj = objects[hit.id];
    Material material = materials[obj.material_id];
    //ambient
    float intensity = ambient * material.ambient* get_occlusion(hit);
    float shadow;
    Intersect result;
    for (int i = 0; i < LIGHTS_NUM; i++) {
        float3 l = normalize(lights[i].pos - hit.pos);
        //shadow
        shadow = Shadow(hit.pos, l, length(lights[i].pos - hit.pos));
        if (shadow == 0.0) {
            continue;
        }
        float d = length(hit.pos - lights[i].pos);
        float att = lights[i].kc + lights[i].kl*d + lights[i].kq*d*d;
        //diffuse
        att = 1;
        float n_dot_l = max(dot(hit.n, l), 0);
        intensity += lights[i].intensity* material.diffuse*n_dot_l*shadow;///att;
             //specular
        if (material.spec != 0) {
            float3 r = 2*n_dot_l*hit.n - l;
            float r_dot_v = dot(r, -ray_dir);
            if (r_dot_v > 0) {
                intensity += lights[i].intensity*pow(r_dot_v, material.spec)*shadow;//att;
            }
        }
    }
    return intensity;
}

vec4 fog(float dist, vec4 color ) {
    return mix(color,vec4(0.5,0.5,0.6, 1), smoothstep(0.0,1.0,1.0 - exp( -dist*0.5 )/50.0) );
}

vec4 get_plane_color(vec3 point) {
    float k = ambient, d, att;
    vec4 color = texture(Plane, point.xz);
    for (int i = 0; i < LIGHTS_NUM; i++) {
        vec3 l = point - lights[i].pos;
        d = length(l);
        l = normalize(l);
        Intersect hit =  ray_intersection(Ray(lights[i].pos, l));
        if (hit.id == -1) {
            k += lights[i].intensity*(-l.y);///(lights[i].kc + lights[i].kl*d + lights[i].kq*d*d);
        }
    }
    return k*texture(Plane, point.xz);
}

vec4 ray_march(inout Stack_frame frame, float4 prev_ret) {
    if (frame.phase == 0) {
        frame.color = float4(0, 0,0,0);
        frame.hit = ray_intersection(frame.ray);
        if (frame.hit.id == -1) {
            if (frame.ray.dir.y < 0) {
                frame.color = get_plane_color(frame.ray.pos -frame.ray.pos.y/frame.ray.dir.y * frame.ray.dir);
                frame.color = vec4(1, 1, 0, 1);
                //frame.color = texture(Plane, frame.ray.pos.xz -
                  //             frame.ray.pos.y/frame.ray.dir.y * vec2(frame.ray.dir.xz))*ambient;
            }
            else {
                frame.color = texture(Cube, frame.ray.dir);
            }
            //frame.color = texture(Cube, frame.ray.dir);
            esp--;
            if (show_fog)
                return fog(length(frame.hit.pos - frame.ray.pos), frame.color);
            //return vec4(1, 1, 0, 1);
            return frame.color;
        }
        //esp--;
        //return vec4(1, 0, 0, 1);
        frame.id = frame.hit.id;
        frame.material_id = objects[frame.id].material_id;
        if (frame.depth >= max_depth) {
             esp--;
             return objects[frame.id].color*get_shade(frame.hit, frame.ray.dir);
        }
        frame.color += color(frame.hit.pos, frame.hit.id)*get_shade(frame.hit, frame.ray.dir) * (1 - materials[frame.material_id].reflection
                                                                   - materials[frame.material_id].refraction);
        if (materials[frame.material_id].reflection > 0) {
            float3 refl = reflect(frame.ray.dir, frame.hit.n);
            vec3 shift = normalize(frame.ray.pos - frame.hit.pos);
            Ray refl_ray = Ray(frame.hit.pos + 2*eps*shift, refl);
            stack[esp] = Stack_frame(0, refl_ray, float4(0, 0, 0, 0), frame.hit, frame.id, frame.material_id, frame.depth + 1);
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
                eta2 = materials[objects[id].material_id].eta;
            }
            float3 refr = refract(frame.ray.dir, frame.hit.n, eta1/eta2);
            if (refr == float3(0, 0, 0)) {
                esp--;
                return frame.color + objects[frame.id].color * materials[frame.material_id].refraction * get_shade(frame.hit, frame.ray.dir);
            }
            vec3 shift = normalize(-frame.ray.pos + frame.hit.pos);
            Ray refr_ray = Ray(frame.hit.pos + 2*eps*shift, -refr);
            stack[esp] = Stack_frame(0, refr_ray, float4(0, 0, 0, 0), frame.hit, frame.id, frame.material_id, frame.depth + 1);
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


void main(void)
{
    float w = float(g_screenWidth);
    float h = float(g_screenHeight);
  // get curr pixelcoordinates
  //
    float x = fragmentTexCoord.x*w;
    float y = fragmentTexCoord.y*h;

  // generate initial ray
  //
    float3 ray_pos = float3(0,0,0);
    float3 ray_dir = EyeRayDir(x,y,w,h, vec2(.5, .5));

  // transorm ray with matrix
  //
    ray_pos = (g_rayMatrix*float4(ray_pos,1)).xyz;
    ray_dir = normalize(float3x3(g_rayMatrix)*ray_dir);
    //fragColor = texture(Cube, ray_dir);
    //return;

  // intersect bounding box of the whole scene, if no intersection found return background color
    Stack_frame zero_frame;
    zero_frame.phase = 0;
    zero_frame.ray = Ray(ray_pos, ray_dir);
    zero_frame.depth = 0;
    stack[0] = zero_frame;
    fragColor = float4(0, 0, 0, 0);
    while (esp != 0) {
        fragColor = ray_march(stack[esp - 1], fragColor);
    }
//    fragColor = res_color;
//    return;
//    float tmin = 1e38f;
//    float tmax = 0;
//
//  if(!RayBoxIntersection(ray_pos, ray_dir, g_bBoxMin, g_bBoxMax, tmin, tmax))
//  {
//    fragColor = g_bgColor;
//    return;
//  }
//
//	float alpha = 1.0f;
//	float3 color = RayMarchConstantFog(tmin, tmax, alpha);
//	fragColor = float4(color,0)*(1.0f-alpha) + g_bgColor*alpha;
//}
}


