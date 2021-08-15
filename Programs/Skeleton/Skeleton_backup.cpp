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
// Nev    : 
// Neptun : 
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

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 450
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 450
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Sphere {
		vec3 center;
		float radius;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	
	};

	struct Ray {
		vec3 start, dir;
	};

	struct Face {
		vec3 vPoints[5];
	};
	
	struct Dodecahedron {
		Face faces[12];
	};

	const int nMaxObjects = 500;
	const int nMaxDodecas = 100;

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[2];  // diffuse, specular, ambient ref
	
	uniform int nObjects;
	uniform Sphere objects[nMaxObjects];

	uniform int nDodecas;
	uniform Dodecahedron dodecas[nMaxDodecas];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
	
	bool zero;
	Hit intersect(const Sphere object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		if(zero){
			hit.mat = 0;
			zero = false;
		}
		else{
			hit.mat = 1;
			zero = true;
		}
		vec3 dist = ray.start - object.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - object.radius * object.radius;
		
		
	
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - object.center) / object.radius;
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nObjects; o++) {
			Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int o = 0; o < nObjects; o++) if (intersect(objects[o], ray).t > 0) return true; //  hit.t < 0 if no intersection
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 5;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} else return outRadiance;
		}
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

//---------------------------
struct Material {
	//---------------------------
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

//---------------------------
struct RoughMaterial : Material {
	//---------------------------
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

vec3 operator/(vec3 u, vec3 v) {
	return vec3(u.x / v.x, u.y / v.y, u.z / v.z);
}

//---------------------------
struct SmoothMaterial : Material {
	//---------------------------
	SmoothMaterial(vec3 n, vec3 k) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + k * k) / ((n + one) * (n + one) + k * k);
		rough = false;
		reflective = true;
	}
};

struct Face {
	vec3 vPoints[5];
	
	Face() = default;

	Face(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec3 v4) {
		vPoints[0] = v0;
		vPoints[1] = v1;
		vPoints[2] = v2;
		vPoints[3] = v3;
		vPoints[4] = v4;
	}
};

struct Dodecahedron {
	
	Face faces[12];

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

		faces[0] = Face(v[0], v[1], v[15], v[4], v[12]);
		faces[1] = Face(v[0], v[12], v[8], v[9], v[13]);
		faces[2] = Face(v[0], v[13], v[5], v[14], v[1]);
		faces[3] = Face(v[1], v[14], v[10], v[11], v[15]);
		faces[4] = Face(v[2], v[3], v[17], v[7], v[16]);
		faces[5] = Face(v[2], v[16], v[11], v[10], v[19]);
		faces[6] = Face(v[2], v[19], v[6], v[18], v[3]);
		faces[7] = Face(v[18], v[9], v[8], v[17], v[3]);
		faces[8] = Face(v[15], v[11], v[16], v[7], v[4]);
		faces[9] = Face(v[4], v[7], v[17], v[8], v[12]);
		faces[10] = Face(v[13], v[9], v[18], v[6], v[5]);
		faces[11] = Face(v[5], v[6], v[19], v[10], v[14]);
	}
};

//---------------------------
struct Sphere {
	//---------------------------
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius) { center = _center; radius = _radius; }
};

//---------------------------
struct Camera {
	//---------------------------
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

//---------------------------
struct Light {
	//---------------------------
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
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
		setUniform(light->direction, "light.direction");
	}

	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}

	void setUniformObjects(const std::vector<Sphere*>& objects) {
		setUniform((int)objects.size(), "nObjects");
		char name[256];
		for (unsigned int o = 0; o < objects.size(); o++) {
			sprintf(name, "objects[%d].center", o);  setUniform(objects[o]->center, name);
			sprintf(name, "objects[%d].radius", o);  setUniform(objects[o]->radius, name);
		}
	}

	void setUniformDodecas(const std::vector<Dodecahedron*>& dodecas) {
		
		setUniform((int)dodecas.size(), "nDodecas");
		char name[256];
		for (unsigned int d = 0; d < dodecas.size(); d++) {
			for (unsigned int f = 0; f < 12; f++) {
				for (unsigned int v = 0; v < 5; v++) {
					sprintf(name, "dodecas[%d].faces[%d].vPoints[%d]", d, f, v);
					setUniform(dodecas[d]->faces[f].vPoints[v], name);
				}
			}
		}
	}
	
};

float rnd() { return (float)rand() / RAND_MAX; }

//---------------------------
class Scene {
	//---------------------------
	std::vector<Sphere*> objects;
	std::vector<Dodecahedron*> dodecas;
	std::vector<Light*> lights;
	Camera camera;
	std::vector<Material*> materials;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(1, 1, 1), vec3(3, 3, 3), vec3(0.4f, 0.3f, 0.3f)));

		vec3 kd(0.3f, 0.2f, 0.7f), ks(10, 10, 10);
		materials.push_back(new RoughMaterial(kd, ks, 100000));

		vec3 n(0.17f, 0.35f, 1.5f), k(3.1f, 2.7f, 1.9f);
		materials.push_back(new SmoothMaterial(n, k));

		objects.push_back(new Sphere(vec3(0, 0, 0), 0.1f));
		objects.push_back(new Sphere(vec3(0, 0, 0.3f), 0.1f));
		dodecas.push_back(new Dodecahedron());
	}

	void setUniform(Shader& shader) {
		shader.setUniformObjects(objects);
		shader.setUniformMaterials(materials);
		shader.setUniformLight(lights[0]);
		shader.setUniformCamera(camera);
		shader.setUniformDodecas(dodecas);
	}

	void Animate(float dt) { camera.Animate(dt); }
};

Shader shader; // vertex and fragment shaders
Scene scene;

//---------------------------
class FullScreenTexturedQuad {
	//---------------------------
	unsigned int vao = 0;	// vertex array object id and texture id
public:
	void create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();

	// create program for the GPU
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
}

// Window has become invalid: Redraw
void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("%d msec\r", (tEnd - tStart) / nFrames);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	scene.setUniform(shader);
	fullScreenTexturedQuad.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.01f);
	glutPostRedisplay();
}