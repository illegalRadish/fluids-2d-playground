/**
 * WebGL2 2D stable fluids (semi-Lagrangian advection + pressure projection).
 * Uses half-float render targets; requires EXT_color_buffer_float.
 */

const VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const err = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(err || "shader compile failed");
  }
  return sh;
}

function program(gl, vertSrc, fragSrc) {
  const vs = compile(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragSrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const err = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(err || "program link failed");
  }
  return p;
}

function createFBO(gl, w, h, internalFormat, format, type, minFilter, magFilter) {
  gl.activeTexture(gl.TEXTURE0);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(tex);
    gl.deleteFramebuffer(fbo);
    throw new Error("framebuffer incomplete: " + status);
  }
  return { texture: tex, fbo, width: w, height: h };
}

/** Velocity splat uses dye stroke exponent × this (legacy 5900/1500 split). */
const STROKE_VEL_SCALE = 5900 / 1500;

function deleteFBO(gl, fbo) {
  if (!fbo) return;
  gl.deleteTexture(fbo.texture);
  gl.deleteFramebuffer(fbo.fbo);
}

export class FluidSim {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error("WebGL2 not available");

    const extFloat = gl.getExtension("EXT_color_buffer_float");
    if (!extFloat) throw new Error("EXT_color_buffer_float required for float render targets");

    gl.getExtension("OES_texture_float_linear");

    this.gl = gl;
    /** Simulation grid width/height (velocity / pressure). */
    this.simSize = 256;
    this.pressureIterations = 20;
    /** Per-frame retention at ~60fps reference (0–1). */
    this.velocityDissipation = 0.96;
    this.dyeDissipation = 0.99;
    /** Scale for pointer delta → velocity splat impulse. */
    this.splatForce = 1600;
    /**
     * Gaussian exponent for dye splat (`exp(-d²·r)`); velocity uses `strokeSize * STROKE_VEL_SCALE`.
     * Higher = tighter / smaller stroke.
     */
    this.strokeSize = 1500;
    /** Vorticity confinement strength (CURL in Pavel's WebGL fluid sim). */
    this.vorticityCurl = 16;
    /** Bloom threshold (bright pass), soft knee, final intensity (Pavel-style). */
    this.bloomThreshold = 0.16;
    this.bloomSoftKnee = 0.39;
    this.bloomIntensity = 1.5;

    this._quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this._quadVAO);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this._programs = this._buildPrograms(gl);

    this._resize();
    this._lastTime = performance.now();
  }

  /**
   * Recreate velocity / divergence / pressure targets after `simSize` changes.
   * Dye buffers (canvas-sized) are unchanged.
   */
  _rebuildSimBuffers() {
    const gl = this.gl;
    const sim = this.simSize;
    if (!this._velRead) return;

    const internalFormat = gl.RGBA16F;
    const format = gl.RGBA;
    const type = gl.HALF_FLOAT;

    deleteFBO(gl, this._velRead);
    deleteFBO(gl, this._velWrite);
    deleteFBO(gl, this._divergence);
    deleteFBO(gl, this._pressureRead);
    deleteFBO(gl, this._pressureWrite);
    deleteFBO(gl, this._curlFBO);

    this._velRead = createFBO(gl, sim, sim, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._velWrite = createFBO(gl, sim, sim, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._divergence = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);
    this._pressureRead = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);
    this._pressureWrite = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);
    this._curlFBO = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);

    this._texelVel = [1 / sim, 1 / sim];
  }

  /** Clamp to 64–512 in steps of 32; rebuilds sim buffers. */
  setSimSize(size) {
    const n = Math.max(64, Math.min(512, Math.round(size / 32) * 32));
    if (n === this.simSize) return;
    this.simSize = n;
    this._rebuildSimBuffers();
  }

  /** Clear velocity and dye to black. */
  clear() {
    const gl = this.gl;
    const clearFbo = (fbo, w, h) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    };
    clearFbo(this._velRead.fbo, this._velRead.width, this._velRead.height);
    clearFbo(this._velWrite.fbo, this._velWrite.width, this._velWrite.height);
    clearFbo(this._dyeRead.fbo, this._dyeRead.width, this._dyeRead.height);
    clearFbo(this._dyeWrite.fbo, this._dyeWrite.width, this._dyeWrite.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _buildPrograms(gl) {
    const advect = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 uTexel;
uniform float uDt;
uniform float uDissipation;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  vec2 coord = vUv - uDt * vel * uTexel;
  fragColor = uDissipation * texture(uSource, coord);
  fragColor.a = 1.0;
}
`
    );

    const divergence = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 fragColor;
void main() {
  float L = texture(uVelocity, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uVelocity, vUv + vec2(uTexel.x, 0.0)).x;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexel.y)).y;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexel.y)).y;
  float div = 0.5 * (R - L + T - B);
  fragColor = vec4(div, 0.0, 0.0, 1.0);
}
`
    );

    const pressure = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 fragColor;
void main() {
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float d = texture(uDivergence, vUv).x;
  float p = (L + R + T + B - d) * 0.25;
  fragColor = vec4(p, 0.0, 0.0, 1.0);
}
`
    );

    const gradientSubtract = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform sampler2D uPressure;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 fragColor;
void main() {
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  vec2 vel = texture(uVelocity, vUv).xy;
  vel -= vec2(R - L, T - B);
  fragColor = vec4(vel, 0.0, 1.0);
}
`
    );

    const curl = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 fragColor;
void main() {
  float L = texture(uVelocity, vUv - vec2(uTexel.x, 0.0)).y;
  float R = texture(uVelocity, vUv + vec2(uTexel.x, 0.0)).y;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexel.y)).x;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexel.y)).x;
  float vorticity = R - L - T + B;
  fragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}
`
    );

    const vorticity = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform vec2 uTexel;
uniform float uCurlStrength;
uniform float uDt;
in vec2 vUv;
out vec4 fragColor;
void main() {
  float L = texture(uCurl, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uCurl, vUv + vec2(uTexel.x, 0.0)).x;
  float T = texture(uCurl, vUv + vec2(0.0, uTexel.y)).x;
  float B = texture(uCurl, vUv - vec2(0.0, uTexel.y)).x;
  float C = texture(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= uCurlStrength * C;
  force.y *= -1.0;
  vec2 vel = texture(uVelocity, vUv).xy;
  vel += force * uDt;
  vel = clamp(vel, -1000.0, 1000.0);
  fragColor = vec4(vel, 0.0, 1.0);
}
`
    );

    const boundary = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  if (vUv.x < uTexel.x || vUv.x > 1.0 - uTexel.x ||
      vUv.y < uTexel.y || vUv.y > 1.0 - uTexel.y) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    fragColor = vec4(vel, 0.0, 1.0);
  }
}
`
    );

    const splatVelocity = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uVelocity;
uniform vec2 uPoint;
uniform vec2 uDelta;
uniform float uRadius;
uniform float uAspect;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec2 p = vUv - uPoint;
  p.x *= uAspect;
  float d = dot(p, p);
  float s = exp(-d * uRadius);
  vec2 base = texture(uVelocity, vUv).xy;
  fragColor = vec4(base + s * uDelta, 0.0, 1.0);
}
`
    );

    const splatDye = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uDye;
uniform vec2 uPoint;
uniform vec3 uColor;
uniform float uRadius;
uniform float uAspect;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec2 p = vUv - uPoint;
  p.x *= uAspect;
  float d = dot(p, p);
  float s = exp(-d * uRadius);
  vec3 base = texture(uDye, vUv).rgb;
  fragColor = vec4(base + s * uColor, 1.0);
}
`
    );

    const displayScene = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uDye;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 c = texture(uDye, vUv).rgb;
  float peak = max(max(c.r, c.g), c.b);
  c = c / (1.0 + peak * 0.42);
  fragColor = vec4(c, 1.0);
}
`
    );

    const bloomPrefilter = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform vec3 uCurve;
uniform float uThreshold;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 c = texture(uScene, vUv).rgb;
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - uCurve.x, 0.0, uCurve.y);
  rq = uCurve.z * rq * rq;
  c *= max(rq, br - uThreshold) / max(br, 0.0001);
  fragColor = vec4(c, 0.0);
}
`
    );

    const bloomBlur = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform vec2 uDirection;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec2 off = uTexel * uDirection;
  vec3 s = texture(uTex, vUv).rgb * 0.227027;
  s += (texture(uTex, vUv + off).rgb + texture(uTex, vUv - off).rgb) * 0.1945946;
  s += (texture(uTex, vUv + off * 2.0).rgb + texture(uTex, vUv - off * 2.0).rgb) * 0.1216216;
  s += (texture(uTex, vUv + off * 3.0).rgb + texture(uTex, vUv - off * 3.0).rgb) * 0.054054;
  s += (texture(uTex, vUv + off * 4.0).rgb + texture(uTex, vUv - off * 4.0).rgb) * 0.016216;
  fragColor = vec4(s, 1.0);
}
`
    );

    const composite = program(
      gl,
      VERT,
      `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomIntensity;
in vec2 vUv;
out vec4 fragColor;
vec3 linearToGamma(vec3 color) {
  color = max(color, vec3(0.0));
  return max(1.055 * pow(color, vec3(0.416666667)) - 0.055, vec3(0.0));
}
void main() {
  vec3 scene = texture(uScene, vUv).rgb;
  vec3 bloom = texture(uBloom, vUv).rgb;
  bloom = linearToGamma(bloom);
  vec3 c = scene + bloom * uBloomIntensity;
  float a = max(c.r, max(c.g, c.b));
  c = c / (1.0 + a * 0.35);
  fragColor = vec4(c, 1.0);
}
`
    );

    return {
      advect,
      divergence,
      pressure,
      gradientSubtract,
      curl,
      vorticity,
      boundary,
      splatVelocity,
      splatDye,
      displayScene,
      bloomPrefilter,
      bloomBlur,
      composite,
    };
  }

  _resize() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this._velRead) return;

    this.canvas.width = w;
    this.canvas.height = h;

    const sim = this.simSize;
    const dyeW = Math.min(1024, w);
    const dyeH = Math.min(1024, h);

    const internalFormat = gl.RGBA16F;
    const format = gl.RGBA;
    const type = gl.HALF_FLOAT;

    deleteFBO(gl, this._velRead);
    deleteFBO(gl, this._velWrite);
    deleteFBO(gl, this._dyeRead);
    deleteFBO(gl, this._dyeWrite);
    deleteFBO(gl, this._divergence);
    deleteFBO(gl, this._pressureRead);
    deleteFBO(gl, this._pressureWrite);
    deleteFBO(gl, this._curlFBO);
    deleteFBO(gl, this._sceneFBO);
    deleteFBO(gl, this._bloomHalf);
    deleteFBO(gl, this._bloomHalf2);

    this._velRead = createFBO(gl, sim, sim, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._velWrite = createFBO(gl, sim, sim, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._dyeRead = createFBO(gl, dyeW, dyeH, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._dyeWrite = createFBO(gl, dyeW, dyeH, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._divergence = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);
    this._pressureRead = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);
    this._pressureWrite = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);
    this._curlFBO = createFBO(gl, sim, sim, internalFormat, format, type, gl.NEAREST, gl.NEAREST);

    const hw = Math.max(1, Math.floor(w / 2));
    const hh = Math.max(1, Math.floor(h / 2));
    this._sceneFBO = createFBO(gl, w, h, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._bloomHalf = createFBO(gl, hw, hh, internalFormat, format, type, gl.LINEAR, gl.LINEAR);
    this._bloomHalf2 = createFBO(gl, hw, hh, internalFormat, format, type, gl.LINEAR, gl.LINEAR);

    this._texelVel = [1 / sim, 1 / sim];
  }

  _bindQuad() {
    this.gl.bindVertexArray(this._quadVAO);
  }

  _blit(target, w, h, draw) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, w, h);
    this._bindQuad();
    draw();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _setTexelUniform(loc, xy) {
    this.gl.uniform2f(loc, xy[0], xy[1]);
  }

  /**
   * @param {number} x normalized 0–1
   * @param {number} y normalized 0–1 (origin top in screen space — caller passes flipped if needed)
   * @param {number} dx velocity impulse in sim space
   * @param {number} dy
   * @param {[number,number,number]} color rgb
   */
  splat(x, y, dx, dy, color) {
    const gl = this.gl;
    const P = this._programs;

    const aspect = this.canvas.width / this.canvas.height;

    this._blit(this._velWrite.fbo, this._velRead.width, this._velRead.height, () => {
      gl.useProgram(P.splatVelocity);
      gl.uniform1i(gl.getUniformLocation(P.splatVelocity, "uVelocity"), 0);
      gl.uniform2f(gl.getUniformLocation(P.splatVelocity, "uPoint"), x, y);
      gl.uniform2f(
        gl.getUniformLocation(P.splatVelocity, "uDelta"),
        dx * this.splatForce,
        dy * this.splatForce
      );
      gl.uniform1f(
        gl.getUniformLocation(P.splatVelocity, "uRadius"),
        this.strokeSize * STROKE_VEL_SCALE
      );
      gl.uniform1f(gl.getUniformLocation(P.splatVelocity, "uAspect"), aspect);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    const tmpV = this._velRead;
    this._velRead = this._velWrite;
    this._velWrite = tmpV;

    this._blit(this._dyeWrite.fbo, this._dyeRead.width, this._dyeRead.height, () => {
      gl.useProgram(P.splatDye);
      gl.uniform1i(gl.getUniformLocation(P.splatDye, "uDye"), 0);
      gl.uniform2f(gl.getUniformLocation(P.splatDye, "uPoint"), x, y);
      gl.uniform3f(gl.getUniformLocation(P.splatDye, "uColor"), color[0], color[1], color[2]);
      gl.uniform1f(gl.getUniformLocation(P.splatDye, "uRadius"), this.strokeSize);
      gl.uniform1f(gl.getUniformLocation(P.splatDye, "uAspect"), aspect);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._dyeRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    const tmpD = this._dyeRead;
    this._dyeRead = this._dyeWrite;
    this._dyeWrite = tmpD;
  }

  step(dt) {
    const gl = this.gl;
    const P = this._programs;
    const dissipationVel = Math.pow(this.velocityDissipation, dt * 60);
    const dissipationDye = Math.pow(this.dyeDissipation, dt * 60);
    const simRes = this.simSize;

    const advectVel = () => {
      this._blit(this._velWrite.fbo, this._velRead.width, this._velRead.height, () => {
        gl.useProgram(P.advect);
        gl.uniform1i(gl.getUniformLocation(P.advect, "uVelocity"), 0);
        gl.uniform1i(gl.getUniformLocation(P.advect, "uSource"), 1);
        gl.uniform2f(gl.getUniformLocation(P.advect, "uTexel"), this._texelVel[0], this._texelVel[1]);
        gl.uniform1f(gl.getUniformLocation(P.advect, "uDt"), dt);
        gl.uniform1f(gl.getUniformLocation(P.advect, "uDissipation"), dissipationVel);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      });
      const t = this._velRead;
      this._velRead = this._velWrite;
      this._velWrite = t;
    };

    this._blit(this._curlFBO.fbo, simRes, simRes, () => {
      gl.useProgram(P.curl);
      gl.uniform1i(gl.getUniformLocation(P.curl, "uVelocity"), 0);
      this._setTexelUniform(gl.getUniformLocation(P.curl, "uTexel"), this._texelVel);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    this._blit(this._velWrite.fbo, this._velRead.width, this._velRead.height, () => {
      gl.useProgram(P.vorticity);
      gl.uniform1i(gl.getUniformLocation(P.vorticity, "uVelocity"), 0);
      gl.uniform1i(gl.getUniformLocation(P.vorticity, "uCurl"), 1);
      gl.uniform1f(gl.getUniformLocation(P.vorticity, "uCurlStrength"), this.vorticityCurl);
      gl.uniform1f(gl.getUniformLocation(P.vorticity, "uDt"), dt);
      this._setTexelUniform(gl.getUniformLocation(P.vorticity, "uTexel"), this._texelVel);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this._curlFBO.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    {
      const t = this._velRead;
      this._velRead = this._velWrite;
      this._velWrite = t;
    }

    this._blit(this._divergence.fbo, this._velRead.width, this._velRead.height, () => {
      gl.useProgram(P.divergence);
      gl.uniform1i(gl.getUniformLocation(P.divergence, "uVelocity"), 0);
      this._setTexelUniform(gl.getUniformLocation(P.divergence, "uTexel"), this._texelVel);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    gl.disable(gl.BLEND);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._pressureRead.fbo);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([0, 0, 0, 0]));
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._pressureWrite.fbo);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([0, 0, 0, 0]));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    for (let i = 0; i < this.pressureIterations; i++) {
      const readP = i % 2 === 0 ? this._pressureRead : this._pressureWrite;
      const writeP = i % 2 === 0 ? this._pressureWrite : this._pressureRead;
      this._blit(writeP.fbo, simRes, simRes, () => {
        gl.useProgram(P.pressure);
        gl.uniform1i(gl.getUniformLocation(P.pressure, "uPressure"), 0);
        gl.uniform1i(gl.getUniformLocation(P.pressure, "uDivergence"), 1);
        this._setTexelUniform(gl.getUniformLocation(P.pressure, "uTexel"), this._texelVel);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, readP.texture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this._divergence.texture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      });
    }

    const pressureOut =
      this.pressureIterations % 2 === 0 ? this._pressureRead : this._pressureWrite;

    this._blit(this._velWrite.fbo, this._velRead.width, this._velRead.height, () => {
      gl.useProgram(P.gradientSubtract);
      gl.uniform1i(gl.getUniformLocation(P.gradientSubtract, "uVelocity"), 0);
      gl.uniform1i(gl.getUniformLocation(P.gradientSubtract, "uPressure"), 1);
      this._setTexelUniform(gl.getUniformLocation(P.gradientSubtract, "uTexel"), this._texelVel);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, pressureOut.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    {
      const t = this._velRead;
      this._velRead = this._velWrite;
      this._velWrite = t;
    }

    this._blit(this._velWrite.fbo, this._velRead.width, this._velRead.height, () => {
      gl.useProgram(P.boundary);
      gl.uniform1i(gl.getUniformLocation(P.boundary, "uVelocity"), 0);
      this._setTexelUniform(gl.getUniformLocation(P.boundary, "uTexel"), this._texelVel);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    {
      const t = this._velRead;
      this._velRead = this._velWrite;
      this._velWrite = t;
    }

    advectVel();

    this._blit(this._velWrite.fbo, this._velRead.width, this._velRead.height, () => {
      gl.useProgram(P.boundary);
      gl.uniform1i(gl.getUniformLocation(P.boundary, "uVelocity"), 0);
      this._setTexelUniform(gl.getUniformLocation(P.boundary, "uTexel"), this._texelVel);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    {
      const t = this._velRead;
      this._velRead = this._velWrite;
      this._velWrite = t;
    }

    this._blit(this._dyeWrite.fbo, this._dyeRead.width, this._dyeRead.height, () => {
      gl.useProgram(P.advect);
      gl.uniform1i(gl.getUniformLocation(P.advect, "uVelocity"), 0);
      gl.uniform1i(gl.getUniformLocation(P.advect, "uSource"), 1);
      this._setTexelUniform(gl.getUniformLocation(P.advect, "uTexel"), this._texelVel);
      gl.uniform1f(gl.getUniformLocation(P.advect, "uDt"), dt);
      gl.uniform1f(gl.getUniformLocation(P.advect, "uDissipation"), dissipationDye);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velRead.texture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this._dyeRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });
    {
      const t = this._dyeRead;
      this._dyeRead = this._dyeWrite;
      this._dyeWrite = t;
    }
  }

  render() {
    const gl = this.gl;
    const P = this._programs;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const bh = this._bloomHalf;
    const bh2 = this._bloomHalf2;

    this._blit(this._sceneFBO.fbo, w, h, () => {
      gl.useProgram(P.displayScene);
      gl.uniform1i(gl.getUniformLocation(P.displayScene, "uDye"), 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._dyeRead.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    const knee = this.bloomThreshold * this.bloomSoftKnee + 0.0001;
    const curve0 = this.bloomThreshold - knee;
    const curve1 = knee * 2.0;
    const curve2 = 0.25 / knee;
    this._blit(bh.fbo, bh.width, bh.height, () => {
      gl.useProgram(P.bloomPrefilter);
      gl.uniform1i(gl.getUniformLocation(P.bloomPrefilter, "uScene"), 0);
      gl.uniform3f(gl.getUniformLocation(P.bloomPrefilter, "uCurve"), curve0, curve1, curve2);
      gl.uniform1f(gl.getUniformLocation(P.bloomPrefilter, "uThreshold"), this.bloomThreshold);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._sceneFBO.texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    const blurH = (src, dst) => {
      this._blit(dst.fbo, dst.width, dst.height, () => {
        gl.useProgram(P.bloomBlur);
        gl.uniform1i(gl.getUniformLocation(P.bloomBlur, "uTex"), 0);
        gl.uniform2f(gl.getUniformLocation(P.bloomBlur, "uTexel"), 1 / src.width, 1 / src.height);
        gl.uniform2f(gl.getUniformLocation(P.bloomBlur, "uDirection"), 1, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, src.texture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      });
    };
    const blurV = (src, dst) => {
      this._blit(dst.fbo, dst.width, dst.height, () => {
        gl.useProgram(P.bloomBlur);
        gl.uniform1i(gl.getUniformLocation(P.bloomBlur, "uTex"), 0);
        gl.uniform2f(gl.getUniformLocation(P.bloomBlur, "uTexel"), 1 / src.width, 1 / src.height);
        gl.uniform2f(gl.getUniformLocation(P.bloomBlur, "uDirection"), 0, 1);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, src.texture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      });
    };

    for (let i = 0; i < 3; i++) {
      blurH(bh, bh2);
      blurV(bh2, bh);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    this._bindQuad();
    gl.useProgram(P.composite);
    gl.uniform1i(gl.getUniformLocation(P.composite, "uScene"), 0);
    gl.uniform1i(gl.getUniformLocation(P.composite, "uBloom"), 1);
    gl.uniform1f(gl.getUniformLocation(P.composite, "uBloomIntensity"), this.bloomIntensity);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._sceneFBO.texture);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bh.texture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.activeTexture(gl.TEXTURE0);
  }

  frame() {
    this._resize();
    const now = performance.now();
    const dt = Math.min(0.05, (now - this._lastTime) / 1000);
    this._lastTime = now;
    this.step(dt);
    this.render();
  }
}
