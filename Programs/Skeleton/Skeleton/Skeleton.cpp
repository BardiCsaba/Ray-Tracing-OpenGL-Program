//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bárdi Csaba
// Neptun : BH9HDV
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          

	layout(location = 0) in vec2 cCamWindowVertex;	
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 pos;
		vec3 Le, La;
	};

	struct Sphere {
		float a, b, c;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	
	};

	struct Ray {
		vec3 start, dir;
	};

	struct DodecFace {
		vec3 vPoints[5];
	};

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[3];
	
	uniform Sphere object;
	uniform DodecFace faces[12];

	in  vec3 p;					
	out vec4 fragmentColor;		
	
	bool inDodeca(vec3 v[5], vec3 normal, vec3 pos){
		float dots[5];
        dots[0] = dot(cross(v[1] - v[0], pos - v[0]), normal);
        dots[1] = dot(cross(v[2] - v[1], pos - v[1]), normal);
        dots[2] = dot(cross(v[3] - v[2], pos - v[2]), normal);
        dots[3] = dot(cross(v[4] - v[3], pos - v[3]), normal);
        dots[4] = dot(cross(v[0] - v[4], pos - v[4]), normal);       
        int negative = 0;
        int positive = 0;
        for(int i = 0; i < 5; i++){
            if(dots[i] < 0)
                negative++;
            else
                positive++;
        }

        if(positive != 5 && negative != 5){
            return false;
		}
		else
			return true;	
	}
		
	Hit intersect_Dodeca(const DodecFace face, Ray ray) {
        Hit hit;
        hit.t = -1;
        hit.mat = 0;

        vec3 normal = cross(face.vPoints[2] - face.vPoints[0], face.vPoints[1] - face.vPoints[0]);
        float t = (dot(face.vPoints[0], normal) - dot(ray.start, normal)) / dot(ray.dir, normal);
        vec3 pos = ray.start + ray.dir * t;

		vec3 corner[5];
        for(int i = 0; i < 5; i++){
            vec3 v = vec3(0,0,0) - face.vPoints[i];
            corner[i] = face.vPoints[i] + v * 0.1f;
        }		
       
        if(inDodeca(face.vPoints,normal, pos)){
            hit.t = t;
			hit.position = pos;
			hit.normal = normal;
		}

		if(inDodeca(corner, normal, pos)){
			hit.t = t;
			hit.position = pos;
			hit.normal = normalize(normal);
			hit.mat = 2;
		}
        return hit;
    }

	bool inSphere(vec3 p){
		return p.x * p.x + p.y * p.y + p.z * p.z < 0.3 * 0.3;
	}


	Hit intersect(const Sphere object, const Ray ray) {
		
		Hit hit;
		hit.t = -1;
		hit.mat = 1;
		
		float a = object.a;
		float b = object.b;
		float c = object.c;
			
		float A = (a * ray.dir.x * ray.dir.x) + (b * ray.dir.y * ray.dir.y); 
		float B = (2.0f * a * ray.start.x * ray.dir.x) + (2.0f * b * ray.start.y * ray.dir.y) - (c * ray.dir.z);
		float C = (a * ray.start.x * ray.start.x) + (b * ray.start.y * ray.start.y) - (c * ray.start.z);
	
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	
		float t2 = (-B - sqrt_discr) / 2.0f / A;
        if (t1 <= 0) return hit;
	
		vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2;
		
		if(!inSphere(p1) && !inSphere(p2))
			return hit;

		if(inSphere(p1) && !inSphere(p2)){
			hit.t = t1;
			hit.position = p1;
		}

		if(!inSphere(p1) && inSphere(p2)){
			hit.t = t2;
			hit.position = p2;
		}

		if(inSphere(p1) && inSphere(p2)){
			if(t1 < t2) {
				hit.t = t1;
				hit.position = p1;			
		}
		else {
			hit.t = t2;			
			hit.position = p2;
			}			
		}

		vec3 u = normalize(vec3(1, 0, 2.0f * a * hit.position.x / c));
		vec3 v = normalize(vec3(0, 1, 2.0f * b * hit.position.y / c));
		
		hit.normal = cross(u,v);
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;

		Hit hit = intersect(object, ray);
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		

		for (int f = 0; f < 12; f++) {
			Hit hit = intersect_Dodeca(faces[f], ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	vec4 QxQ(vec4 q1, vec4 q2){
		vec4 q;
		q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
		q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
		q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
		q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
		return q;
	}
	
	vec4 createQuat(vec3 axis, float angle){
		vec4 q;
		q.x = axis.x * sin(angle / 2);
		q.y = axis.y * sin(angle / 2);
		q.z = axis.z * sin(angle / 2);
		q.w = cos(angle / 2);
		return q;
	}

	vec3 rotate(vec3 point, vec3 axis, float angle){
		vec4 q = createQuat(axis, angle);
		vec3 v = point;
		vec4 qv = QxQ(q, vec4(v, 0));
		return QxQ(qv, vec4(-q.x, -q.y, -q.z, q.w)).xyz; 
	}	

	float distance(vec3 p, vec3 q){
		return sqrt(abs(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2)));
	}

	const float epsilon = 0.001f;
	const int maxdepth = 5;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) break;
	
			if (materials[hit.mat].rough == 1) {
				vec3 lightdir = normalize(light.pos - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				float dis = distance(hit.position, light.pos);
				if (cosTheta > 0) {
					vec3 LeIn = light.Le / dot(light.pos - hit.position, light.pos - hit.position);
					outRadiance = weight * LeIn * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(-hit.normal, halfway);
					if (cosDelta > 0) 
						outRadiance = weight * LeIn * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);		
				}
				weight *= materials[hit.mat].ka / dis;
				break;
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, 1.0 - dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}

			if(materials[hit.mat].reflective == 0 && materials[hit.mat].rough == 0){
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
				
				ray.start = rotate(ray.start, hit.normal, 3.1415f / 2.5f);
				ray.dir = rotate(ray.dir, hit.normal, 3.1415f / 2.5f);
			}
		}
		outRadiance += weight * light.La;
		return outRadiance;
	}

	void main() {

		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

struct Material {
	
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

struct RoughMaterial : Material {
	
	RoughMaterial(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) {
		ka = _ka;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

struct PortalMaterial : Material {
	
	PortalMaterial() {
		rough = false;
		reflective = false;
	}
};

vec3 operator/(vec3 u, vec3 v) {
	return vec3(u.x / v.x, u.y / v.y, u.z / v.z);
}

struct SmoothMaterial : Material {
	
	SmoothMaterial(vec3 n, vec3 k) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + k * k) / ((n + one) * (n + one) + k * k);
		rough = false;
		reflective = true;
	}
};

struct Sphere {
	
	float a,b,c;

	Sphere(float _a, float _b, float _c) { a = _a; b = _b; c = _c; }
};

struct DodecFace {
	vec3 vPoints[5];

	DodecFace() = default;

	DodecFace(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec3 v4) {
		vPoints[0] = v0;
		vPoints[1] = v1;
		vPoints[2] = v2;
		vPoints[3] = v3;
		vPoints[4] = v4;
	}
};

struct Dodecahedron {

	DodecFace faces[12];

	Dodecahedron() {
		vec3 v[20];
		v[0] = vec3(0, 0.618034, 1.61803);
		v[1] = vec3(0, -0.618034, 1.61803);
		v[2] = vec3(0, -0.618034, -1.61803);
		v[3] = vec3(0, 0.618034, -1.61803);
		v[4] = vec3(1.61803, 0, 0.618034);
		v[5] = vec3(-1.61803, 0, 0.618034);
		v[6] = vec3(-1.61803, 0, -0.618034);
		v[7] = vec3(1.61803, 0, -0.618034);
		v[8] = vec3(0.618034, 1.61803, 0);
		v[9] = vec3(-0.618034, 1.61803, 0);
		v[10] = vec3(-0.618034, -1.61803, 0);
		v[11] = vec3(0.618034, -1.61803, 0);
		v[12] = vec3(1, 1, 1);
		v[13] = vec3(-1, 1, 1);
		v[14] = vec3(-1, -1, 1);
		v[15] = vec3(1, -1, 1);
		v[16] = vec3(1, -1, -1);
		v[17] = vec3(1, 1, -1);
		v[18] = vec3(-1, 1, -1);
		v[19] = vec3(-1, -1, -1);

		faces[0] = DodecFace(v[0], v[1], v[15], v[4], v[12]);
		faces[1] = DodecFace(v[0], v[12], v[8], v[9], v[13]);
		faces[2] = DodecFace(v[0], v[13], v[5], v[14], v[1]);
		faces[3] = DodecFace(v[1], v[14], v[10], v[11], v[15]);
		faces[4] = DodecFace(v[2], v[3], v[17], v[7], v[16]);
		faces[5] = DodecFace(v[2], v[16], v[11], v[10], v[19]);
		faces[6] = DodecFace(v[2], v[19], v[6], v[18], v[3]);
		faces[7] = DodecFace(v[18], v[9], v[8], v[17], v[3]);
		faces[8] = DodecFace(v[15], v[11], v[16], v[7], v[4]);
		faces[9] = DodecFace(v[4], v[7], v[17], v[8], v[12]);
		faces[10] = DodecFace(v[13], v[9], v[18], v[6], v[5]);
		faces[11] = DodecFace(v[5], v[6], v[19], v[10], v[14]);
	}
};

struct Camera {
	
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};

struct Light {
	
	vec3 pos;
	vec3 Le, La;
	Light(vec3 _pos, vec3 _Le, vec3 _La) {
		pos = _pos;
		Le = _Le; La = _La;
	}
};

class Shader : public GPUProgram {

public:
	void setUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (unsigned int mat = 0; mat < materials.size(); mat++) {
			sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name); 
			sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
			sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
			sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
			sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
			sprintf(name, "materials[%d].rough", mat); setUniform(materials[mat]->rough, name);
			sprintf(name, "materials[%d].reflective", mat); setUniform(materials[mat]->reflective, name);
		}
	}

	void setUniformLight(Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->Le, "light.Le");
		setUniform(light->pos, "light.pos");
	}

	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}

	void setUniformObjects(const Sphere* object) {

		setUniform(object->a, "object.a");
		setUniform(object->b, "object.b");
		setUniform(object->c, "object.c");
	}
	void setUniformDodeca(const Dodecahedron dodeca) {
		
		char name[256];
		for (unsigned int f = 0; f < 12; f++) {
			for (unsigned int v = 0; v < 5; v++) {
				sprintf(name, "faces[%d].vPoints[%d]", f, v );
				setUniform(dodeca.faces[f].vPoints[v], name);
			}
		}
		
	}
};

class Scene {
	
	std::vector<Sphere*> objects;
	std::vector<Light*> lights;
	std::vector<Material*> materials;
	Camera camera;
	Dodecahedron dodeca;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1.3f);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(0, 0, 0.3f),vec3(3,3,3), vec3(0.5f, 0.8f, 0.9f)));

		vec3 kd(0.7f, 0.3f, 0.3f), ks(10, 10, 10);
		vec3 ka = kd * 3;
		materials.push_back(new RoughMaterial(ka, kd, ks, 10));
		
		vec3 n(0.17f, 0.35f, 1.5f), k(3.1f, 2.7f, 1.9f);
		materials.push_back(new SmoothMaterial(n, k));

		materials.push_back(new PortalMaterial());

		objects.push_back(new Sphere(0.8f, 1.4f, 0.4f));
	}

	void setUniform(Shader& shader) {
		shader.setUniformObjects(objects[0]);
		shader.setUniformMaterials(materials);
		shader.setUniformLight(lights[0]);
		shader.setUniformCamera(camera);
		shader.setUniformDodeca(dodeca);
	}

	void Animate(float dt) { camera.Animate(dt); }
};

Shader shader;
Scene scene;

class FullScreenTexturedQuad {
	
	unsigned int vao = 0;	
public:
	void create() {
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	

		
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	  
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);    
	}

	void Draw() {
		glBindVertexArray(vao);	
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();

	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
}

void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("%d msec\r", (tEnd - tStart) / nFrames);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	scene.setUniform(shader);
	fullScreenTexturedQuad.Draw();

	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	scene.Animate(0.01f);
	glutPostRedisplay();
}