import OpenGL.GL as gl
from OpenGL.GL import shaders
import numpy as np
import os

__all__ = ['Shader']

class Shader:

    def __init__(self, vertex_str, fragment_str, geometry_str=None, uniform_data={}):
        if geometry_str is not None:
            self.ID = shaders.compileProgram(
                shaders.compileShader(self._load(vertex_str), gl.GL_VERTEX_SHADER),
                shaders.compileShader(self._load(fragment_str), gl.GL_FRAGMENT_SHADER),
                shaders.compileShader(self._load(geometry_str), gl.GL_GEOMETRY_SHADER),
            )
        else:
            self.ID = shaders.compileProgram(
                shaders.compileShader(self._load(vertex_str), gl.GL_VERTEX_SHADER),
                shaders.compileShader(self._load(fragment_str), gl.GL_FRAGMENT_SHADER),
            )

        self.uniform_data = uniform_data
        self._in_use = False

    def _load(self, shader_source):
        if os.path.exists(shader_source):
            with open(shader_source, 'r') as shader_file:
                shader_source = shader_file.readlines()
        return shader_source

    def set_uniform(self, name, data, type:str):
        self.uniform_data[name] = (data, type)
        # if self._in_use:
        #     self.__set_uniform(name, data, type)

    def use(self):
        gl.glUseProgram(self.ID)
        self._in_use = True
        try: ## load uniform values into program
            for key, data in self.uniform_data.items():
                self.__set_uniform(key, *data)
        except:
            gl.glUseProgram(0)
            raise

    def unuse(self):
        self._in_use = False
        gl.glUseProgram(0)

    def __enter__(self):
        self.use()

    def __exit__(self, exc_type, exc_value, traceback):
        self._in_use = False
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}: {exc_value}")
        gl.glUseProgram(0)

    def __set_uniform(self, name, value, type:str, cnt=1):
        if type in ["bool", "int", "sampler2D"]:
            gl.glUniform1iv(gl.glGetUniformLocation(self.ID, name), cnt, np.array(value, dtype=np.int32))
        elif type == "float":
            gl.glUniform1fv(gl.glGetUniformLocation(self.ID, name), cnt, np.array(value, dtype=np.float32))
        elif type == "vec2":
            gl.glUniform2fv(gl.glGetUniformLocation(self.ID, name), cnt, np.array(value, dtype=np.float32))
        elif type == "vec3":
            gl.glUniform3fv(gl.glGetUniformLocation(self.ID, name), cnt, np.array(value, dtype=np.float32))
        elif type == "vec4":
            gl.glUniform4fv(gl.glGetUniformLocation(self.ID, name), cnt, np.array(value, dtype=np.float32))
        elif type == "mat3":
            gl.glUniformMatrix3fv(gl.glGetUniformLocation(self.ID, name), cnt, gl.GL_FALSE, np.array(value, dtype=np.float32))
        elif type == "mat4":
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.ID, name), cnt, gl.GL_FALSE, np.array(value, dtype=np.float32))

    def delete(self):
        print("delete shader")
        gl.glDeleteProgram(self.ID)