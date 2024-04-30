//
//  sphere_scene.c
//  Rasterizer
//
//
#include "color.h"
#include "ray.h"
#include "vec3.h"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
unsigned char* Image1 = new unsigned char [512 * 512 * 3] ();
unsigned char* Image2 = new unsigned char [512 * 512 * 3] ();
unsigned char* Image3 = new unsigned char [512 * 512 * 3] ();
unsigned char* Image4 = new unsigned char [512 * 512 * 3] ();
#define M_PI 3.14159265358979323846

int     gNumVertices    = 0;    // Number of 3D vertices.
int     gNumTriangles   = 0;    // Number of triangles.
int*    gIndexBuffer    = NULL; // Vertex indices for the triangles.

vec3* vertices = NULL;
vec3* vertices_ = NULL;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void MatrixMutiply(float result[4][4], const float mat1[4][4], const float mat2[4][4]){
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            result[i][j] = 0.0;
            for (int k = 0; k < 4; k++){
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

void MatrixMultiply1(float result[4][1], const float mat1[4][4], const float mat2[4][1]){
    for (int i = 0; i < 4; i++){
        result[i][0] = 0.0;
        for (int j = 0; j < 4; j++){
            result[i][0] += mat1[i][j] * mat2[j][0];
        }
    }
}

void MatrixNorm(float result[4][1]){
    float w = result[3][0];
    for(int i = 0; i <4; i++){
        result[i][0] /= w;
    }
}

class Sphere{
public:
    vec3 o;
    double r;
};

class light{
public:
    vec3 o;
};

class k_ads{
public:
    vec3 a;
    vec3 d;
    vec3 s;
    double s_p;
};

vec3 cal_L(k_ads k, int I, vec3 n, vec3 l, vec3 v){
    vec3 La = k.a * 0.2;
    vec3 Ld = k.d * I * std::max(0.0, dot(n, l));
    vec3 Ls = k.s * I * pow(std::max(0.0, dot(n, (v + l) / (v + l).length())), k.s_p);
    vec3 L = La + Ld + Ls;
    return L;
}

vec3 crossProduct(vec3 A, vec3 B){
    vec3 result;
    float a = A.y()*B.z() - A.z()*B.y();
    float b = A.z()*B.x() - A.x()*B.z();
    float c = A.x()*B.y() - A.y()*B.x();
    result = vec3{a, b, c};
    return result;
}

vec3* create_scene(float Mcam[][4], float Mm[][4])
{
    int width   = 32;
    int height  = 16;
    
    float theta, phi;
    int t;
    float result4x4[4][4];
    float result4x5[4][4];
    float result4x6[4][4];
    float result4x7[4][4];
    float result4x1[4][1];

    MatrixMutiply(result4x4, Mcam, Mm);

    gNumVertices    = (height - 2) * width + 2;
    gNumTriangles   = (height - 2) * (width - 1) * 2;

    vertices = new vec3[gNumVertices];

    gIndexBuffer    = new int[3*gNumTriangles];
    
    t = 0;
    for (int j = 1; j < height-1; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            theta = (float) j / (height-1) * M_PI;
            phi   = (float) i / (width-1)  * M_PI * 2;
            
            float   x   = sinf(theta) * cosf(phi);
            float   y   = cosf(theta);
            float   z   = -sinf(theta) * sinf(phi);

            float p0[4][1] = {{x}, {y}, {z}, {1}};

            MatrixMultiply1(result4x1, result4x4, p0);
            MatrixNorm(result4x1);
            
            vertices[t] = vec3(result4x1[0][0], result4x1[1][0], result4x1[2][0]);
            
            t++;
        }
    }

    float p0[4][1] = {{0.0}, {1.0}, {0.0}, {1.0}};

    MatrixMultiply1(result4x1, result4x4, p0);
    MatrixNorm(result4x1);
            
    vertices[t] = vec3(result4x1[0][0], result4x1[1][0], result4x1[2][0]);
            
    t++;
    
    float p1[4][1] = {{0.0}, {-1.0}, {0.0}, {1.0}};

    MatrixMultiply1(result4x1, result4x4, p1);
    MatrixNorm(result4x1);

    vertices[t] = vec3(result4x1[0][0], result4x1[1][0], result4x1[2][0]);
            
    t++;
    
    t = 0;
    for (int j = 0; j < height-3; ++j)
    {
        for (int i = 0; i < width-1; ++i)
        {
            gIndexBuffer[t++] = j*width + i;
            gIndexBuffer[t++] = (j+1)*width + (i+1);
            gIndexBuffer[t++] = j*width + (i+1);
            gIndexBuffer[t++] = j*width + i;
            gIndexBuffer[t++] = (j+1)*width + i;
            gIndexBuffer[t++] = (j+1)*width + (i+1);
        }
    }
    for (int i = 0; i < width-1; ++i)
    {
        gIndexBuffer[t++] = (height-2)*width;
        gIndexBuffer[t++] = i;
        gIndexBuffer[t++] = i + 1;
        gIndexBuffer[t++] = (height-2)*width + 1;
        gIndexBuffer[t++] = (height-3)*width + (i+1);
        gIndexBuffer[t++] = (height-3)*width + i;
    }

    return vertices;
}

vec3* create_scene_(float Mvp[][4], float Morth[][4], float P[][4], float Mcam[][4], float Mm[][4])
{
    int width   = 32;
    int height  = 16;
    
    float theta, phi;
    int t;
    float result4x4[4][4];
    float result4x5[4][4];
    float result4x6[4][4];
    float result4x7[4][4];
    float result4x1[4][1];

    MatrixMutiply(result4x4, Mvp, Morth);
    MatrixMutiply(result4x5, result4x4, P);
    MatrixMutiply(result4x6, result4x5, Mcam);
    MatrixMutiply(result4x7, result4x6, Mm);

    gNumVertices    = (height - 2) * width + 2;
    gNumTriangles   = (height - 2) * (width - 1) * 2;

    vertices_ = new vec3[gNumVertices];

    gIndexBuffer    = new int[3*gNumTriangles];
    
    t = 0;
    for (int j = 1; j < height-1; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            theta = (float) j / (height-1) * M_PI;
            phi   = (float) i / (width-1)  * M_PI * 2;
            
            float   x   = sinf(theta) * cosf(phi);
            float   y   = cosf(theta);
            float   z   = -sinf(theta) * sinf(phi);

            float p0[4][1] = {{x}, {y}, {z}, {1}};

            MatrixMultiply1(result4x1, result4x7, p0);
            MatrixNorm(result4x1);
            
            vertices_[t] = vec3(result4x1[0][0], result4x1[1][0], result4x1[2][0]);
            
            t++;
        }
    }
    
    float p0[4][1] = {{0.0}, {1.0}, {0.0}, {1.0}};

    MatrixMultiply1(result4x1, result4x7, p0);
    MatrixNorm(result4x1);

    vertices_[t] = vec3(result4x1[0][0], result4x1[1][0], result4x1[2][0]);
            
    t++;
    
    float p1[4][1] = {{0.0}, {-1.0}, {0.0}, {1.0}};


    MatrixMultiply1(result4x1, result4x7, p1);
    MatrixNorm(result4x1);
            
    vertices_[t] = vec3(result4x1[0][0], result4x1[1][0], result4x1[2][0]);
            
    t++;
    
    t = 0;
    for (int j = 0; j < height-3; ++j)
    {
        for (int i = 0; i < width-1; ++i)
        {
            gIndexBuffer[t++] = j*width + i;
            gIndexBuffer[t++] = (j+1)*width + (i+1);
            gIndexBuffer[t++] = j*width + (i+1);
            gIndexBuffer[t++] = j*width + i;
            gIndexBuffer[t++] = (j+1)*width + i;
            gIndexBuffer[t++] = (j+1)*width + (i+1);
        }
    }
    for (int i = 0; i < width-1; ++i)
    {
        gIndexBuffer[t++] = (height-2)*width;
        gIndexBuffer[t++] = i;
        gIndexBuffer[t++] = i + 1;
        gIndexBuffer[t++] = (height-2)*width + 1;
        gIndexBuffer[t++] = (height-3)*width + (i+1);
        gIndexBuffer[t++] = (height-3)*width + i;
    }
    return vertices_;
}

int main(int argc, char* argv[])
{
    float n_x = 512;
    float n_y = 512;
    float l = -0.1;
    float r = 0.1;
    float b = -0.1;
    float t = 0.1;
    float n = -0.1;
    float f = -1000;

    Sphere sphere = {vec3(0.0, 0.0, -7.0), 2.0};

    float Mvp[4][4]{
        {n_x/2, 0, 0, (n_x - 1)/2},
        {0, n_y/2, 0, (n_y-1)/2},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    float Morth[4][4] = {
        {2/(r-l), 0, 0, -(r+l)/(r-l)},
        {0, 2/(t-b), 0, -(t+b)/(t-b)},
        {0, 0, 2/(n-f), -(n+f)/(n-f)},
        {0, 0, 0, 1}
    };
    float P[4][4] = {
        {n, 0, 0, 0},
        {0, n, 0, 0},
        {0, 0, (n+f), -(n*f)},
        {0, 0, 1, 0}
    };
    float Mcam[4][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    float Mm[4][4] = {
        {2, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 2, -7},
        {0, 0, 0, 1}    
    };

    k_ads k_s = {
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.5, 0.0),
        vec3(0.5, 0.5, 0.5),
        32
    };

    float gamma_ = 2.2f;

    vec3 light = {vec3(-4.0, 4.0, -3.0)};

    vec3 c_p;

    float Mcam_Mm[4][4];

    create_scene(Mcam, Mm);
    create_scene_(Mvp, Morth, P, Mcam, Mm);

    vec3 tri1, tri2, tri3;

    vec3 tri1_, tri2_, tri3_;

    vec3 point1, point2, point3;

    for (int i = 0; i < gNumTriangles; i++){
        int k0 = gIndexBuffer[3*i + 0];
        int k1 = gIndexBuffer[3*i + 1];
        int k2 = gIndexBuffer[3*i + 2];

        tri1 = vertices_[k0];
        tri2 = vertices_[k1];
        tri3 = vertices_[k2];

        int x_min = floor(std::min(std::min(tri1.x(), tri2.x()), tri3.x()));
        int x_max = ceil(std::max(std::max(tri1.x(), tri2.x()), tri3.x()));
        int y_min = floor(std::min(std::min(tri1.y(), tri2.y()), tri3.y()));
        int y_max = ceil(std::max(std::max(tri1.y(), tri2.y()), tri3.y()));

        float beta = ((tri1.y() - tri3.y()) * x_min + (tri3.x() - tri1.x()) * y_min + tri1.x() * tri3.y() - tri3.x() * tri1.y()) / 
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma = ((tri1.y() - tri2.y()) * x_min + (tri2.x() - tri1.x()) * y_min + tri1.x() * tri2.y() - tri2.x() * tri1.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        int n = (x_max - x_min) + 1;

        float beta_x = (tri1.y() - tri3.y()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float beta_y = (tri3.x() - tri1.x()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma_x = (tri1.y() - tri2.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());
        float gamma_y = (tri2.x() - tri1.x()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        for (int y = y_min; y <= y_max; y++){
            for (int x = x_min; x <= x_max; x++){
                if (beta > 0 && gamma > 0 && (beta + gamma) < 1){
                    int index = (y * n_x + x) * 3;
                    Image1[index] = 255;
                    Image1[index + 1] = 255;
                    Image1[index + 2] = 255;
                }
                beta += beta_x;
                gamma += gamma_x;
            }
            beta += beta_y - n * beta_x;
            gamma += gamma_y - n * gamma_x;
        }
    }

    std::vector<float> depthBuffer2(n_x * n_y, -INFINITY);

    for (int i = 0; i < gNumTriangles; i++){
        int k0 = gIndexBuffer[3*i + 0];
        int k1 = gIndexBuffer[3*i + 1];
        int k2 = gIndexBuffer[3*i + 2];

        tri1 = vertices[k0];
        tri2 = vertices[k1];
        tri3 = vertices[k2];

        vec3 a = tri2 - tri1;
        vec3 b = tri3 - tri1;

        c_p = (tri1 + tri2 + tri3) / 3;

        vec3 v = (-c_p) / (-c_p).length();
        vec3 l = (light - c_p) / (light - c_p).length();
        vec3 n = (crossProduct(a, b)) / (crossProduct(a, b)).length();

        vec3 color_s = cal_L(k_s, 1, n, l, v);

        tri1 = vertices_[k0];
        tri2 = vertices_[k1];
        tri3 = vertices_[k2];

        int x_min = floor(std::min(std::min(tri1.x(), tri2.x()), tri3.x()));
        int x_max = ceil(std::max(std::max(tri1.x(), tri2.x()), tri3.x()));
        int y_min = floor(std::min(std::min(tri1.y(), tri2.y()), tri3.y()));
        int y_max = ceil(std::max(std::max(tri1.y(), tri2.y()), tri3.y()));

        float beta = ((tri1.y() - tri3.y()) * x_min + (tri3.x() - tri1.x()) * y_min + tri1.x() * tri3.y() - tri3.x() * tri1.y()) / 
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma = ((tri1.y() - tri2.y()) * x_min + (tri2.x() - tri1.x()) * y_min + tri1.x() * tri2.y() - tri2.x() * tri1.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        int num = (x_max - x_min) + 1;

        float beta_x = (tri1.y() - tri3.y()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float beta_y = (tri3.x() - tri1.x()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma_x = (tri1.y() - tri2.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());
        float gamma_y = (tri2.x() - tri1.x()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        for (int y = y_min; y <= y_max; y++){
            for (int x = x_min; x <= x_max; x++){
                if (beta > 0 && gamma > 0 && (beta + gamma) < 1){
                    float depth = (1-(beta+gamma)) * tri1.z() + beta * tri2.z() + gamma * tri3.z();
                    if(depth > depthBuffer2[y * n_x + x]){
                        depthBuffer2[y * n_x + x] = depth;
                        int index = (y * n_x + x) * 3;
                        Image2[index] = pow(color_s.x() > 1 ? 1 : color_s.x(), 1/gamma_) * 255;
                        Image2[index + 1] = pow(color_s.y() > 1 ? 1 : color_s.y(), 1/gamma_) * 255;
                        Image2[index + 2] = pow(color_s.z() > 1 ? 1 : color_s.z(), 1/gamma_) * 255;
                    }
                }
                beta += beta_x;
                gamma += gamma_x;
            }
            beta += beta_y - num * beta_x;
            gamma += gamma_y - num * gamma_x;
        }
    }

    std::vector<float> depthBuffer3(n_x * n_y, -INFINITY);

    for (int i = 0; i < gNumTriangles; i++){
        int k0 = gIndexBuffer[3*i + 0];
        int k1 = gIndexBuffer[3*i + 1];
        int k2 = gIndexBuffer[3*i + 2];

        tri1 = vertices[k0];
        tri2 = vertices[k1];
        tri3 = vertices[k2];

        vec3 v1 = (-tri1) / (-tri1).length();
        vec3 v2 = (-tri2) / (-tri2).length();
        vec3 v3 = (-tri3) / (-tri3).length();
        vec3 l1 = (light - tri1) / (light - tri1).length();
        vec3 l2 = (light - tri2) / (light - tri2).length();
        vec3 l3 = (light - tri3) / (light - tri3).length();
        vec3 n1 = (tri1 - sphere.o) / (tri1 - sphere.o).length();
        vec3 n2 = (tri2 - sphere.o) / (tri2 - sphere.o).length();
        vec3 n3 = (tri3 - sphere.o) / (tri3 - sphere.o).length();

        vec3 color_tri1 = cal_L(k_s, 1, n1, l1, v1);
        vec3 color_tri2 = cal_L(k_s, 1, n2, l2, v2);
        vec3 color_tri3 = cal_L(k_s, 1, n3, l3, v3);

        tri1 = vertices_[k0];
        tri2 = vertices_[k1];
        tri3 = vertices_[k2];

        int x_min = floor(std::min(std::min(tri1.x(), tri2.x()), tri3.x()));
        int x_max = ceil(std::max(std::max(tri1.x(), tri2.x()), tri3.x()));
        int y_min = floor(std::min(std::min(tri1.y(), tri2.y()), tri3.y()));
        int y_max = ceil(std::max(std::max(tri1.y(), tri2.y()), tri3.y()));

        float beta = ((tri1.y() - tri3.y()) * x_min + (tri3.x() - tri1.x()) * y_min + tri1.x() * tri3.y() - tri3.x() * tri1.y()) / 
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma = ((tri1.y() - tri2.y()) * x_min + (tri2.x() - tri1.x()) * y_min + tri1.x() * tri2.y() - tri2.x() * tri1.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        int num = (x_max - x_min) + 1;

        float beta_x = (tri1.y() - tri3.y()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float beta_y = (tri3.x() - tri1.x()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma_x = (tri1.y() - tri2.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());
        float gamma_y = (tri2.x() - tri1.x()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        for (int y = y_min; y <= y_max; y++){
            for (int x = x_min; x <= x_max; x++){
                if (beta > 0 && gamma > 0 && (beta + gamma) < 1){
                    float depth = (1-(beta+gamma)) * tri1.z() + beta * tri2.z() + gamma * tri3.z();
                    if(depth > depthBuffer3[y * n_x + x]){
                        depthBuffer3[y * n_x + x] = depth;
                        vec3 color_s = (1- (beta + gamma)) * color_tri1 + beta * color_tri2 + gamma * color_tri3;

                        int index = (y * n_x + x) * 3;
                        Image3[index] = pow(color_s.x() > 1 ? 1 : color_s.x(), 1/gamma_) * 255;
                        Image3[index + 1] = pow(color_s.y() > 1 ? 1 : color_s.y(), 1/gamma_) * 255;
                        Image3[index + 2] = pow(color_s.z() > 1 ? 1 : color_s.z(), 1/gamma_) * 255;
                    }
                }
                beta += beta_x;
                gamma += gamma_x;
            }
            beta += beta_y - num * beta_x;
            gamma += gamma_y - num * gamma_x;
        }
    }

    std::vector<float> depthBuffer4(n_x * n_y, -INFINITY);

    for (int i = 0; i < gNumTriangles; i++){
        int k0 = gIndexBuffer[3*i + 0];
        int k1 = gIndexBuffer[3*i + 1];
        int k2 = gIndexBuffer[3*i + 2];

        tri1_ = vertices[k0];
        tri2_ = vertices[k1];
        tri3_ = vertices[k2];

        vec3 n1 = (tri1_ - sphere.o) / (tri1_ - sphere.o).length();
        vec3 n2 = (tri2_ - sphere.o) / (tri2_ - sphere.o).length();
        vec3 n3 = (tri3_ - sphere.o) / (tri3_ - sphere.o).length();

        vec3 tri1 = vertices_[k0];
        vec3 tri2 = vertices_[k1];
        vec3 tri3 = vertices_[k2];

        int x_min = floor(std::min(std::min(tri1.x(), tri2.x()), tri3.x()));
        int x_max = ceil(std::max(std::max(tri1.x(), tri2.x()), tri3.x()));
        int y_min = floor(std::min(std::min(tri1.y(), tri2.y()), tri3.y()));
        int y_max = ceil(std::max(std::max(tri1.y(), tri2.y()), tri3.y()));

        float beta = ((tri1.y() - tri3.y()) * x_min + (tri3.x() - tri1.x()) * y_min + tri1.x() * tri3.y() - tri3.x() * tri1.y()) / 
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma = ((tri1.y() - tri2.y()) * x_min + (tri2.x() - tri1.x()) * y_min + tri1.x() * tri2.y() - tri2.x() * tri1.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        int num = (x_max - x_min) + 1;

        float beta_x = (tri1.y() - tri3.y()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float beta_y = (tri3.x() - tri1.x()) /
                    ((tri1.y() - tri3.y()) * tri2.x() + (tri3.x() - tri1.x()) * tri2.y() + tri1.x() * tri3.y() - tri3.x() * tri1.y());
        float gamma_x = (tri1.y() - tri2.y()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());
        float gamma_y = (tri2.x() - tri1.x()) /
                    ((tri1.y() - tri2.y()) * tri3.x() + (tri2.x() - tri1.x()) * tri3.y() + tri1.x() * tri2.y() - tri2.x() * tri1.y());

        for (int y = y_min; y <= y_max; y++){
            for (int x = x_min; x <= x_max; x++){
                if (beta > 0 && gamma > 0 && (beta + gamma) < 1){
                    float depth = (1-(beta+gamma)) * tri1.z() + beta * tri2.z() + gamma * tri3.z();
                    vec3 point = (1-(beta+gamma)) * tri1_ + beta * tri2_+ gamma * tri3_;
                    vec3 norm = ((1-(beta+gamma)) * n1 + beta * n2 + gamma * n3) / ((1-(beta+gamma)) * n1 + beta * n2 + gamma * n3).length();
                    vec3 v = (-point) / (-point).length();
                    vec3 l = (light - point) / (light - point).length();
                    vec3 color_phong = cal_L(k_s, 1, norm, l, v);
                    if(depth > depthBuffer4[y * n_x + x]){
                        depthBuffer4[y * n_x + x] = depth;

                        int index = (y * n_x + x) * 3;
                        Image4[index] = pow(color_phong.x() > 1 ? 1 : color_phong.x(), 1/gamma_) * 255;
                        Image4[index + 1] = pow(color_phong.y() > 1 ? 1 : color_phong.y(), 1/gamma_) * 255;
                        Image4[index + 2] = pow(color_phong.z() > 1 ? 1 : color_phong.z(), 1/gamma_) * 255;
                    }
                }
                beta += beta_x;
                gamma += gamma_x;
            }
            beta += beta_y - num * beta_x;
            gamma += gamma_y - num * gamma_x;
        }
    }

    glfwInit();

    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window1 = glfwCreateWindow(n_x, n_y, "unshaded", NULL, NULL);
    if(!window1){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    GLFWwindow* window2 = glfwCreateWindow(n_x, n_y, "flat_shading", NULL, NULL);
    if(!window2){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    GLFWwindow* window3 = glfwCreateWindow(n_x, n_y, "gouraud_shading", NULL, NULL);
    if(!window2){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    GLFWwindow* window4 = glfwCreateWindow(n_x, n_y, "phong_shading", NULL, NULL);
    if(!window2){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window1);
    glfwMakeContextCurrent(window2);
    glfwMakeContextCurrent(window3);
    glfwMakeContextCurrent(window4);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, n_x, n_y);
    glfwSetFramebufferSizeCallback(window1, framebuffer_size_callback);
    glfwSetFramebufferSizeCallback(window2, framebuffer_size_callback);
    glfwSetFramebufferSizeCallback(window3, framebuffer_size_callback);
    glfwSetFramebufferSizeCallback(window4, framebuffer_size_callback);
    
    while(!glfwWindowShouldClose(window1) && !glfwWindowShouldClose(window2) && !glfwWindowShouldClose(window3) && !glfwWindowShouldClose(window4))
    {
        glfwMakeContextCurrent(window1);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glad_glDrawPixels(n_x, n_y, GL_RGB, GL_UNSIGNED_BYTE, Image1);
    
        glfwSwapBuffers(window1);
        glfwPollEvents();

        glfwMakeContextCurrent(window2);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glad_glDrawPixels(n_x, n_y, GL_RGB, GL_UNSIGNED_BYTE, Image2);
    
        glfwSwapBuffers(window2);
        glfwPollEvents();

        glfwMakeContextCurrent(window3);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glad_glDrawPixels(n_x, n_y, GL_RGB, GL_UNSIGNED_BYTE, Image3);
    
        glfwSwapBuffers(window3);
        glfwPollEvents();

        glfwMakeContextCurrent(window4);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glad_glDrawPixels(n_x, n_y, GL_RGB, GL_UNSIGNED_BYTE, Image4);
    
        glfwSwapBuffers(window4);
        glfwPollEvents();
    };

    glfwTerminate();
    return 0;
}