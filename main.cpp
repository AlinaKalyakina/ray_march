//internal includes
#include "common.h"
#include "ShaderProgram.h"
#include "LiteMath.h"

//External dependencies
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <random>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static GLsizei WIDTH = 512, HEIGHT = 512; //размеры окна

using namespace LiteMath;

float3 g_camPos(0, 1, 3);//
float3 view_dir(0, 0, -1);
float  cam_rot[2] = {0,0};
int    mx = 0, my = 0;
float speed = 0.1;
const float max_speed = 4;
//const float init_speed = 0.1;
//const float acceleration = 1.5;
bool soft_shadows = false;
bool fog = false;
std::string tex_path = "../tex/";

std::string textures[] = {"ft.tga", "bk.tga",
                          "up.tga", "dn.tga",
                          "rt.tga", "lf.tga"};


void windowResize(GLFWwindow* window, int width, int height) {
  WIDTH  = width;
  HEIGHT = height;
}

static void mouseMove(GLFWwindow* window, double xpos, double ypos) {
  xpos *= 0.05f;
  ypos *= 0.05f;

  int x1 = int(xpos);
  int y1 = int(ypos);

  cam_rot[1] += 0.05f*(x1 - mx);	//Изменение угола поворота
  cam_rot[0] -= 0.05f*(y1 - my);
  view_dir.x = cos(cam_rot[0])*sin(cam_rot[1]);
  view_dir.y = sin(cam_rot[0]);
  view_dir.z = -cos(cam_rot[0])*cos(cam_rot[1]);

  mx = int(xpos);
  my = int(ypos);
}

static void mouseScroll(GLFWwindow* window, double xpos, double ypos) {
    g_camPos += ypos*view_dir*0.1;
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if (action == GLFW_RELEASE && !(key == GLFW_KEY_RIGHT_SHIFT || key == GLFW_KEY_LEFT_SHIFT)) {
        return;
    }
    switch (key) {
        case GLFW_KEY_W:
            g_camPos += speed*view_dir;
            break;
        case GLFW_KEY_S:
            g_camPos -= speed*view_dir;
            break;
        case GLFW_KEY_D:
            g_camPos -= speed*float3(view_dir.z, 0, -view_dir.x);
            break;
        case GLFW_KEY_A:
            g_camPos += speed*float3(view_dir.z, 0, -view_dir.x);
            break;
        case GLFW_KEY_Q:
            cam_rot[1] -= speed;
            break;
        case GLFW_KEY_E:
            cam_rot[1]+= speed;
            break;
        case GLFW_KEY_R:
            cam_rot[0] += speed;
            break;
        case GLFW_KEY_F:
            cam_rot[0] -= speed;
            break;
        case GLFW_KEY_0:
            cam_rot[0] = cam_rot[1] = 0;
            g_camPos = float3(0, 0, 3);
            break;
        case GLFW_KEY_T:
            g_camPos.y += speed;
            break;
        case GLFW_KEY_G:
            g_camPos.y -= speed;
            break;
        case GLFW_KEY_1:
            if (action == GLFW_PRESS)
                soft_shadows = ! soft_shadows;
            break;
        case GLFW_KEY_2:
            if (action == GLFW_PRESS)
                fog = ! fog;
            break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            if (speed < max_speed && action == GLFW_PRESS) {
                speed *= 2;
            }
            if (action == GLFW_RELEASE)
                speed = 0.1;
            printf("%lf ", speed);
            break;
    }
    if (g_camPos.y < 0) {
        g_camPos.y = 0;
    }
    view_dir.x = cos(cam_rot[0])*sin(cam_rot[1]);
    view_dir.y = sin(cam_rot[0]);
    view_dir.z = -cos(cam_rot[0])*cos(cam_rot[1]);
}

int initGL() {
	int res = 0;
	//грузим функции opengl через glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	std::cout << "Vendor: "   << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: "  << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL: "     << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	return 0;
}

int main(int argc, char** argv)
{
	if(!glfwInit())
    return -1;
//
	//запрашиваем контекст opengl версии 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); 
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE); 

  GLFWwindow*  window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL ray marching sample", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
    double x, y;
	glfwGetCursorPos(window, &x, &y);
	mx = int(0.05*x);
	my = int(0.05*y);
    //glfwSetCursorPosCallback (window, mouseMove);
    glfwSetScrollCallback(window, mouseScroll);
    glfwSetWindowSizeCallback(window, windowResize);
    glfwSetKeyCallback(window, keyCallback);

	glfwMakeContextCurrent(window); 
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

	if(initGL() != 0) 
		return -1;
	
  //Reset any OpenGL errors which could be present for some reason
	GLenum gl_error = glGetError();
	while (gl_error != GL_NO_ERROR)
		gl_error = glGetError();

	//создание шейдерной программы из двух файлов с исходниками шейдеров
	//используется класс-обертка ShaderProgram
	std::unordered_map<GLenum, std::string> shaders;
	shaders[GL_VERTEX_SHADER]   = "../shaders/vertex.glsl";
	shaders[GL_FRAGMENT_SHADER] = "../shaders/fragment.glsl";
	ShaderProgram program(shaders); GL_CHECK_ERRORS;//

  glfwSwapInterval(1); // force 60 frames per second
  
  //Создаем и загружаем геометрию поверхности
  //
  GLuint g_vertexBufferObject;
  GLuint g_vertexArrayObject;//
  //
 
    float quadPos[] =
    {
      -1.0f,  1.0f,	// v0 - top left corner
      -1.0f, -1.0f,	// v1 - bottom left corner
      1.0f,  1.0f,	// v2 - top right corner
      1.0f, -1.0f	  // v3 - bottom right corner
    };
    unsigned int cubemap;
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
    int width, height, nrChannels;
    unsigned char* image;
    for (GLuint i = 0; i < 6; i++) {
        image = stbi_load((tex_path + textures[i]).c_str(), &width, &height, &nrChannels, 0);
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
        stbi_image_free(image);
    }
    //glGenerateTextureMipmap(cubemap);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    unsigned int texture;
    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    image = stbi_load((tex_path + "plane.jpg").c_str(), &width, &height, &nrChannels, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    g_vertexBufferObject = 0;
    GLuint vertexLocation = 0; // simple layout, assume have only positions at location = 0

    glGenBuffers(1, &g_vertexBufferObject);                                                        GL_CHECK_ERRORS;
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferObject);                                           GL_CHECK_ERRORS;
    glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), (GLfloat*)quadPos, GL_STATIC_DRAW);     GL_CHECK_ERRORS;

    glGenVertexArrays(1, &g_vertexArrayObject);                                                    GL_CHECK_ERRORS;
    glBindVertexArray(g_vertexArrayObject);                                                        GL_CHECK_ERRORS;

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferObject);                                           GL_CHECK_ERRORS;
    glEnableVertexAttribArray(vertexLocation);                                                     GL_CHECK_ERRORS;
    glVertexAttribPointer(vertexLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);                            GL_CHECK_ERRORS;

    glBindVertexArray(0);

	//цикл обработки сообщений и отрисовки сцены каждый кадр
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		//очищаем экран каждый кадр
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);               GL_CHECK_ERRORS;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); GL_CHECK_ERRORS;

        program.StartUseShader();                          GL_CHECK_ERRORS;

        float4x4 camRotMatrix   = mul(rotate_Y_4x4(-cam_rot[1]), rotate_X_4x4(+cam_rot[0]));
        float4x4 camTransMatrix = translate4x4(g_camPos);
        float4x4 rayMatrix      = mul(camTransMatrix, camRotMatrix);
        program.SetUniform("g_rayMatrix", rayMatrix);
        program.SetUniform("show_fog", fog);
        program.SetUniform("g_screenWidth" , WIDTH);
        program.SetUniform("g_screenHeight", HEIGHT);
        program.SetUniform("show_soft_shadows", soft_shadows);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
        program.SetUniform("Cube", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture);
        program.SetUniform("Plane", 1);
        // очистка и заполнение экрана цвет//
        glViewport  (0, 0, WIDTH, HEIGHT);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // draw calli
    //
    glBindVertexArray(g_vertexArrayObject); GL_CHECK_ERRORS;
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  GL_CHECK_ERRORS;  // The last parameter of glDrawArrays is equal to VS invocations
    
    program.StopUseShader();

		glfwSwapBuffers(window); 
	}

	//очищаем vboи vao перед закрытием программы
  //
	glDeleteVertexArrays(1, &g_vertexArrayObject);
  glDeleteBuffers(1,      &g_vertexBufferObject);

	glfwTerminate();
	return 0;
}
